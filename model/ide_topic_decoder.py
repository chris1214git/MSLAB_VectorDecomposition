import sys
import random
import datetime
import wordcloud
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
# from tqdm.auto import tqdm

sys.path.append("./")
from utils.loss import ListNet, MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.eval_topic import CoherenceNPMI, TopicDiversity, InvertedRBO
from utils.toolbox import get_free_gpu, record_settings
from model.inference_network import ContextualInferenceNetwork

class IDEDataset(Dataset):
    def __init__(self, corpus, emb, target):
        
        assert len(emb) == len(target)
        self.corpus = corpus
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)        
        
    def __getitem__(self, idx):
        return self.corpus[idx], self.emb[idx], self.target[idx]

    def __len__(self):
        return len(self.emb)

class DecoderNetwork(nn.Module):
    def __init__(self, config, device, vocab_size, contextual_size, glove_word_embeddings, n_components, hidden_sizes=(100,100), activation='relu', dropout=0.2, learn_priors=True):
        super(DecoderNetwork, self).__init__()

        assert activation in ['softplus', 'relu']

        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.contextual_size = contextual_size
        self.glove_word_embeddings = glove_word_embeddings
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None

        # decoder architecture
        self.batch_norm = nn.BatchNorm1d(vocab_size)
        self.word_embedding =  nn.Parameter(torch.randn(vocab_size*4, vocab_size))
        ## architecture - parallel
        ## architecture - ratio merge
        self.para_full_decoder_tanh = nn.Sequential(
            nn.Linear(contextual_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        self.para_full_decoder_sigmoid = nn.Sequential(
            nn.Linear(contextual_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        self.compress = nn.Sequential(
            nn.Linear(vocab_size*2, vocab_size),
        )
        ## architecture - concatenate
        self.con_full_decoder_tanh = nn.Sequential(
            nn.Linear(contextual_size+vocab_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        self.con_full_decoder_sigmoid = nn.Sequential(
            nn.Linear(contextual_size+vocab_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        ## architecture - concatenate_word
        self.half_decoder_tanh = nn.Sequential(
            nn.Linear(vocab_size+contextual_size, vocab_size*4),
            nn.BatchNorm1d(vocab_size*4),
            nn.Tanh(),
            nn.Dropout(p=0.2),
        )
        self.half_decoder_sigmoid = nn.Sequential(
            nn.Linear(vocab_size+contextual_size, vocab_size*4),
            nn.BatchNorm1d(vocab_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
        )
        ## architecture - TBD
        self.share_wieght_decoder = nn.Sequential(
            nn.Linear(contextual_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Sigmoid(),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
        )
        self.glove_emb_decoder = nn.Sequential(
            nn.Linear(vocab_size+contextual_size, vocab_size*4),
            nn.BatchNorm1d(vocab_size*4),
            nn.Sigmoid(),
            nn.Linear(vocab_size*4, glove_word_embeddings.shape[1]),
            nn.BatchNorm1d(glove_word_embeddings.shape[1]),
        )
        
        # topic model architecture
        self.inf_net = ContextualInferenceNetwork(vocab_size, contextual_size, n_components, hidden_sizes, activation, label_size=0)
        
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components).to(device)
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor([topic_prior_variance] * n_components).to(device)
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, vocab_size).to(device)
        self.beta = nn.Parameter(self.beta)
        
        nn.init.xavier_uniform_(self.beta)
        
        self.beta_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)
        
        self.drop_theta = nn.Dropout(p=self.dropout)
    
    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, emb, target, labels=None):
        """Forward pass."""
        posterior_mu, posterior_log_sigma = self.inf_net(target, emb, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA
        # in: batch_size x input_size x n_components
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        # word_dist: batch_size x input_size
        word_dist_for_decoder = word_dist.detach()
        self.topic_word_matrix = self.beta
        if self.config['architecture'] == 'concatenate_word':
            if self.config['activation'] == 'tanh':
                emb_word_dist = torch.cat((word_dist_for_decoder, emb), dim=1)
                decoded_word_dist = self.half_decoder_tanh(emb_word_dist)
                recon_dist = self.batch_norm((torch.matmul(decoded_word_dist, self.word_embedding)))
            else:
                emb_word_dist = torch.cat((word_dist_for_decoder, emb), dim=1)
                decoded_word_dist = self.half_decoder_sigmoid(emb_word_dist)
                recon_dist = self.batch_norm((torch.matmul(decoded_word_dist, self.word_embedding)))
        elif self.config['architecture'] == 'concatenate':
            if self.config['activation'] == 'tanh':
                emb_word_dist = torch.cat((word_dist_for_decoder, emb), dim=1)
                decoded_word_dist = self.con_full_decoder_tanh(emb_word_dist)
                recon_dist = decoded_word_dist
            else:
                emb_word_dist = torch.cat((word_dist_for_decoder, emb), dim=1)
                decoded_word_dist = self.con_full_decoder_sigmoid(emb_word_dist)
                recon_dist = decoded_word_dist
        elif self.config['architecture'] == 'parallel':
            if self.config['activation'] == 'tanh':
                emb_word_dist = self.para_full_decoder_tanh(emb)
                decoded_word_dist = torch.cat((word_dist_for_decoder, emb_word_dist), dim=1)
                recon_dist = self.compress(decoded_word_dist)
            else:
                emb_word_dist = self.para_full_decoder_sigmoid(emb)
                decoded_word_dist = torch.cat((word_dist_for_decoder, emb_word_dist), dim=1)
                recon_dist = self.compress(decoded_word_dist)
        elif self.config['architecture'] == 'ratio_merge':
            if self.config['activation'] == 'tanh':
                decoded_word_dist = self.para_full_decoder_tanh(emb)
                recon_dist = 0.8 * decoded_word_dist + 0.2 * word_dist_for_decoder
            else:
                decoded_word_dist = self.para_full_decoder_sigmoid(emb)
                recon_dist = 0.8 * decoded_word_dist + 0.2 * word_dist_for_decoder
        else:
            recon_dist = word_dist_for_decoder
        return self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, recon_dist
    
    def get_theta(self, target, emb, labels=None):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net(target, emb, labels)
            theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta

class IDETopicDecoder:
    def __init__(self, config, texts=None, vocab = None, idx2token=None, device=None, contextual_size=768, word_embeddings=None, 
                n_components=10, hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True,
                momentum=0.99, reduce_on_plateau=False, num_data_loader_workers=mp.cpu_count(), loss_weights=None):
        self.config = config
        self.texts = texts
        self.vocab = vocab
        self.idx2token = idx2token
        self.device = device
        self.contextual_size = contextual_size
        self.word_embeddings = word_embeddings
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.reduce_on_plateau = reduce_on_plateau
        self.momentum = momentum
        self.num_data_loader_workers = num_data_loader_workers
        self.training_doc_topic_distributions = None
        self.distribution_cache = None
        self.num_epochs = config['epochs']
        if config['loss'] == 'mse':
            self.loss_funct = torch.nn.MSELoss(reduction='mean')
        else:
             self.loss_funct = MythNet
        if loss_weights:
            self.weights = loss_weights
        else:
            self.weights = {"beta": 1}

        self.model = DecoderNetwork(
                    config, device, len(vocab), contextual_size, word_embeddings, n_components, hidden_sizes, activation,
                    dropout, learn_priors)
        
        self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])

        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)
        
        self.best_components = None

    def loss(self, inputs, word_dists, recon_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
        logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(dim=1)

        KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)

        RL = torch.sum(-inputs * torch.log(word_dists + 1e-10), dim=1)

        DL = self.loss_funct(recon_dists, inputs)

        return KL, RL, DL

    def training(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch, (corpus, emb, target) in enumerate(loader):
            target = target.reshape(target.shape[0], -1)
            emb, target = emb.to(self.device), target.to(self.device)

            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists, recon_dists = self.model(emb, target)

            kl_loss, rl_loss, dl_loss = self.loss(
                target, word_dists, recon_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)
            loss = self.weights["beta"] * kl_loss + rl_loss + dl_loss
            loss = loss.sum()

            loss.backward()
            self.optimizer.step()

            samples_processed += target.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss
    
    def validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0

        results = defaultdict(list)
        dists = defaultdict(list)

        for batch, (corpus, emb, target) in enumerate(loader):
            target = target.reshape(target.shape[0], -1)
            emb, target = emb.to(self.device), target.to(self.device)

            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists, recon_dists = self.model(emb, target)
            
            kl_loss, rl_loss, dl_loss = self.loss(target, word_dists, recon_dists, prior_mean, prior_variance,
                              posterior_mean, posterior_variance, posterior_log_variance)

            loss = self.weights["beta"] * kl_loss + rl_loss + dl_loss
            loss = loss.sum()

            samples_processed += target.size()[0]
            val_loss += loss.item()

             # Semantic Prcision for reconstruct
            if self.config['semantic']:
                precision_scores, word_result = semantic_precision_all(recon_dists, target, self.word_embeddings, self.vocab, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    results['[Recon] Semantic Precision v1@{}'.format(k)].append(v)

                precision_scores, word_result = semantic_precision_all_v2(recon_dists, target, self.word_embeddings, self.vocab, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    results['[Recon] Semantic Precision v2@{}'.format(k)].append(v)
                
            # Precision for reconstruct
            precision_scores = retrieval_precision_all(recon_dists, target, k=self.config['topk'])
            for k, v in precision_scores.items():
                results['[Recon] Precision v1@{}'.format(k)].append(v)
            
            precision_scores = retrieval_precision_all_v2(recon_dists, target, k=self.config['topk'])
            for k, v in precision_scores.items():
                results['[Recon] Precision v2@{}'.format(k)].append(v)

            # NDCG for reconstruct
            ndcg_scores = retrieval_normalized_dcg_all(recon_dists, target, k=self.config['topk'])
            for k, v in ndcg_scores.items():
                results['[Recon] ndcg@{}'.format(k)].append(v)

            # Semantic Prcision for word dist
            if self.config['semantic']:
                precision_scores, word_result = semantic_precision_all(word_dists, target, self.word_embeddings, self.vocab, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    dists['[Word Dist] Semantic Precision v1@{}'.format(k)].append(v)

                precision_scores, word_result = semantic_precision_all_v2(word_dists, target, self.word_embeddings, self.vocab, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    dists['[Word Dist] Semantic Precision v2@{}'.format(k)].append(v)
                
            # Precision for word dist
            precision_scores = retrieval_precision_all(word_dists, target, k=self.config['topk'])
            for k, v in precision_scores.items():
                dists['[Word Dist] Precision v1@{}'.format(k)].append(v)

            precision_scores = retrieval_precision_all_v2(word_dists, target, k=self.config['topk'])
            for k, v in precision_scores.items():
                dists['[Word Dist] Precision v2@{}'.format(k)].append(v)

            # NDCG for word dist
            ndcg_scores = retrieval_normalized_dcg_all(word_dists, target, k=self.config['topk'])
            for k, v in ndcg_scores.items():
                dists['[Word Dist] ndcg@{}'.format(k)].append(v)
        
        for k in results:
            results[k] = np.mean(results[k])
        
        for k in dists:
            dists[k] = np.mean(dists[k])

        val_loss /= samples_processed

        return samples_processed, val_loss, results, dists
    
    def validation2(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0

        results = defaultdict(list)
        dists = defaultdict(list)

        for batch, (corpus, emb, target) in enumerate(loader):
            target = target.reshape(target.shape[0], -1)
            emb, target = emb.to(self.device), target.to(self.device)

            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists, recon_dists = self.model(emb, target)
            
            for i in range(recon_dists.shape[0]):
                r = recon_dists[i].view(1,-1)
                t = target[i].view(1,-1)
                # Precision for reconstruct
                precision_scores = retrieval_precision_all(r, t, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['Precision@{}'.format(k)].append(v)
                
                precision_scores = retrieval_precision_all_v2(r, t, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['R-Precision@{}'.format(k)].append(v)

                # NDCG for reconstruct
                ndcg_scores = retrieval_normalized_dcg_all(r, t, k=self.config['topk'])
                for k, v in ndcg_scores.items():
                    results['NDCG@{}'.format(k)].append(v)
            
        return results

    def fit(self, training_set, validation_set, n_samples=20):
        self.model.to(self.device)
        train_loader = DataLoader(training_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        validation_loader = DataLoader(validation_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        
        train_loss = 0
        samples_processed = 0
        
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        record_settings(self.config)

        for epoch in range(self.config['epochs']):
            s = datetime.datetime.now()
            sp, train_loss = self.training(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            if  (epoch + 1) % 10 == 0:
                s = datetime.datetime.now()
                val_samples_processed, val_loss, val_res, dist_res = self.validation(validation_loader)
                e = datetime.datetime.now()

                pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(training_set) * self.num_epochs, train_loss, val_loss, e - s))
                
                npmi = CoherenceNPMI(texts=self.texts, topics=self.get_topic_lists(10))
                diversity = InvertedRBO(topics=self.get_topic_lists(10))
                record = open('./'+self.config['experiment']+'_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['architecture']+'_'+self.config['activation']+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_batch'+str(self.config['batch_size'])+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print('EPOCH', epoch + 1)
                record.write('EPOCH '+ str(epoch + 1) + '\n')
                for key,val in val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")
                for key,val in dist_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")
                print('NPMI: ', npmi.score())
                print('IRBO: ', diversity.score())
                record.write('NPMI: '+ str(npmi.score()) + '\n')
                record.write('IRBO: '+ str(diversity.score()) + '\n')

                if (epoch + 1) == 10:
                    recon_df = pd.DataFrame.from_dict(val_res, orient='index').T
                    dist_df = pd.DataFrame.from_dict(dist_res, orient='index').T
                else:
                    recon_df = pd.concat([recon_df, pd.DataFrame.from_dict(val_res, orient='index').T], axis=0)
                    dist_df = pd.concat([dist_df, pd.DataFrame.from_dict(dist_res, orient='index').T], axis=0)

            self.best_components = self.model.beta
            pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch + 1, self.num_epochs, samples_processed,
                len(training_set) * self.num_epochs, train_loss, e - s))
        #recon_df.to_csv('./'+self.config['experiment']+'_recon_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['architecture']+'_'+self.config['activation']+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_batch'+str(self.config['batch_size'])+'_weightdecay'+str(self.config['weight_decay'])+'.csv', index=False)
        #dist_df.to_csv('./'+self.config['experiment']+'_dist_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['architecture']+'_'+self.config['activation']+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_batch'+str(self.config['batch_size'])+'_weightdecay'+str(self.config['weight_decay'])+'.csv', index=False)
        pbar.close()
    
    def get_topic_lists(self, k=10):
        """
        Retrieve the lists of topic words.

        :param k: (int) number of words to return per topic, default 10.
        """
        assert k <= len(self.vocab), "k must be <= input size."
        # TODO: collapse this method with the one that just returns the topics
        component_dists = self.best_components
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics
    def get_doc_topic_distribution(self, dataset, n_samples=20):
        n_samples = self.n_components
        if self.distribution_cache is not None:
            return self.distribution_cache
        self.model.eval()

        loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)
        pbar = tqdm(n_samples, position=0, leave=True)
        final_thetas = []
        for sample_index in range(n_samples):
            with torch.no_grad():
                collect_theta = []

                for batch, (corpus, emb, target) in enumerate(loader):
                    # batch_size x vocab_size
                    emb, target = emb.to(self.device), target.to(self.device)
                    # forward pass
                    self.model.zero_grad()
                    collect_theta.extend(self.model.get_theta(target, emb).cpu().numpy().tolist())

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))
        pbar.close()
        self.distribution_cache = np.sum(final_thetas, axis=0) / n_samples
        return self.distribution_cache
        
    def get_reconstruct(self, testing_set):
        loader = DataLoader(testing_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)
        self.model.eval()
        recon_lists = []
        target_lists = []
        documents_lists = ()
        with torch.no_grad():
            for batch, (corpus, emb, target) in enumerate(loader):     
                target = target.reshape(target.shape[0], -1)
                emb, target = emb.to(self.device), target.to(self.device)

                self.model.zero_grad()
                prior_mean, prior_variance, posterior_mean, posterior_variance,\
                posterior_log_variance, word_dists, recon_dists = self.model(emb, target)
                
                kl_loss, rl_loss, dl_loss = self.loss(target, word_dists, recon_dists, prior_mean, prior_variance,
                                posterior_mean, posterior_variance, posterior_log_variance)
                recon_lists.append(recon_dists)
                target_lists.append(target)
                documents_lists = documents_lists + corpus

        return torch.cat(recon_lists, dim=0).cpu().detach().numpy(), torch.cat(target_lists, dim=0).cpu().detach().numpy(), documents_lists