import torch
from torch import nn
from torch.nn import functional as F

from contextualized_topic_models.networks.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):

    ### casimir
    # (1) Add parameter vocab_size
    def __init__(self, config, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, vocab_size=0, word_embedding=None):
    ###
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.config = config
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None

        ### casimir
        self.vocab_size = vocab_size
        self.batch_norm = nn.BatchNorm1d(vocab_size)
        
        ## random initialize word embedding
        if config['architecture'] == 'after':
            self.word_embedding = nn.Parameter(torch.randn(vocab_size*4+bert_size, vocab_size))
        else:
            self.word_embedding = nn.Parameter(torch.randn(vocab_size*4, vocab_size))
        ## glove word embedding
        # self.word_embedding = nn.Parameter(word_embedding)

        # #concatenated & share weight version
        self.decoder = nn.Sequential(
            nn.Linear(vocab_size, vocab_size*4),
            nn.BatchNorm1d(vocab_size*4),
            nn.Sigmoid(),
            nn.Linear(vocab_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
            nn.Sigmoid(),
        )
        ## doc embedding info version
        if config['architecture'] == 'after':
            if config['activation'] == 'tanh':
                self.half_decoder = nn.Sequential(
                    nn.Linear(vocab_size, vocab_size*4),
                    nn.BatchNorm1d(vocab_size*4),
                    nn.Tanh(),
                    nn.Dropout(p=0.2),
                )
            else:
                self.half_decoder = nn.Sequential(
                    nn.Linear(vocab_size, vocab_size*4),
                    nn.BatchNorm1d(vocab_size*4),
                    nn.Sigmoid(),
                    nn.Dropout(p=0.2),
                )
        else:
            if config['activation'] == 'tanh':
                self.half_decoder = nn.Sequential(
                    nn.Linear(vocab_size+bert_size, vocab_size*4),
                    nn.BatchNorm1d(vocab_size*4),
                    nn.Tanh(),
                    nn.Dropout(p=0.2),
                )
            else:
                self.half_decoder = nn.Sequential(
                    nn.Linear(vocab_size+bert_size, vocab_size*4),
                    nn.BatchNorm1d(vocab_size*4),
                    nn.Sigmoid(),
                    nn.Dropout(p=0.2),
                )
        ## real share weight
        self.real_share_wieght_decoder = nn.Sequential(
            nn.Linear(bert_size, bert_size*4),
            nn.BatchNorm1d(bert_size*4),
            nn.Sigmoid(),
            nn.Linear(bert_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
            nn.Sigmoid(),
        )
        ## use glove embedding version
        self.glove_emb_decoder = nn.Sequential(
            nn.Linear(vocab_size+bert_size, vocab_size*4),
            nn.BatchNorm1d(vocab_size*4),
            nn.Sigmoid(),
            nn.Linear(vocab_size*4, word_embedding.shape[1]),
            nn.BatchNorm1d(word_embedding.shape[1]),
            nn.Sigmoid(),
        )

        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        ## Real share weight version
        # x_bert = self.transform_doc_emb(x_bert)
        # recon_dist = self.real_share_wieght_decoder(x_bert)

        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            # word_dist: batch_size x input_size
            self.topic_word_matrix = self.beta
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")
        
        ### casimir
        ## concatenated version
        # word_dist = self.decoder(word_dist)

        ## share weight version
        # recon_dist = self.decoder(word_dist)

        ## doc embedding info version
        # emb_word_dist = torch.cat((word_dist, x_bert), dim=1)
        # word_dist = torch.sigmoid(self.batch_norm((torch.matmul(emb_word_dist, self.word_embedding))))
        # recon_dist = word_dist

        ## share weight + doc embedding info version
        # decoded_word_dist = self.half_decoder(word_dist)
        # emb_word_dist = torch.cat((decoded_word_dist, x_bert), dim=1)
        # recon_dist = torch.sigmoid(self.batch_norm((torch.matmul(emb_word_dist, self.word_embedding))))
        
        if self.config['architecture'] == 'after':
            decoded_word_dist = self.half_decoder(word_dist)
            emb_word_dist = torch.cat((decoded_word_dist, x_bert), dim=1)
            recon_dist = torch.sigmoid(self.batch_norm((torch.matmul(emb_word_dist, self.word_embedding))))
        else:
            emb_word_dist = torch.cat((word_dist, x_bert), dim=1)
            decoded_word_dist = self.half_decoder(emb_word_dist)
            recon_dist = torch.sigmoid(self.batch_norm((torch.matmul(decoded_word_dist, self.word_embedding))))

        ## share weight + doc embedding info version + glove word embedding
        # emb_word_dist = torch.cat((word_dist, x_bert), dim=1)
        # decoded_word_dist = self.glove_emb_decoder(emb_word_dist)
        # recon_dist = torch.sigmoid(self.batch_norm((torch.matmul(decoded_word_dist, torch.transpose(self.word_embedding, 0, 1)))))
        
        ## ZTM baseline
        # recon_dist = word_dist

        # classify labels

        estimated_labels = None

        if labels is not None:
            estimated_labels = self.label_classification(theta)

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, estimated_labels, recon_dist

    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
