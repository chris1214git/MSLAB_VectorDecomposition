import datetime
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import wordcloud
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextualized_topic_models.utils.early_stopping.early_stopping import EarlyStopping
from contextualized_topic_models.networks.mlp_network import mlp_network
### casimir
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, TopicDiversity, InvertedRBO
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all
###


class MLP:
    """Class to train the contextualized topic model. This is the more general class that we are keeping to
    avoid braking code, users should use the two subclasses ZeroShotTM and CombinedTm to do topic modeling.

    :param bow_size: int, dimension of input
    :param contextual_size: int, dimension of input that comes from BERT embeddings
    :param inference_type: string, you can choose between the contextual model and the combined model
    :param n_components: int, number of topic components, (default 10)
    :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
    :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
    :param activation: string, 'softplus', 'relu', (default 'softplus')
    :param dropout: float, dropout to use (default 0.2)
    :param learn_priors: bool, make priors a learnable parameter (default True)
    :param batch_size: int, size of batch to use for training (default 64)
    :param lr: float, learning rate to use for training (default 2e-3)
    :param momentum: float, momentum to use for training (default 0.99)
    :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
    :param num_epochs: int, number of epochs to train for, (default 100)
    :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
    :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
    :param label_size: int, number of total labels (default: 0)
    :param loss_weights: dict, it contains the name of the weight parameter (key) and the weight (value) for each loss.
    It supports only the weight parameter beta for now. If None, then the weights are set to 1 (default: None).

    """
    ### casimir
    # (1) Add config, text, vocab, idx2token
    def __init__(self, bow_size, contextual_size, batch_size=64, lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_data_loader_workers=mp.cpu_count(), config=None, texts=None, vocab = None, tp_vocab = None, word_embeddings=None, idx2token=None):
    ###
        self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.bow_size = bow_size
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.num_data_loader_workers = num_data_loader_workers
        self.config = config
        self.texts = texts
        self.vocab = vocab
        self.tp_vocab = tp_vocab
        self.word_embeddings = word_embeddings
        self.idx2token = idx2token
        self.distribution_cache = None
        self.train_data = None
        self.validation_data = None
        self.nn_epoch = None

        self.model = mlp_network(contextual_size, len(vocab))

        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)

    def _loss(self, inputs, recon_dists):
        recon_dists = torch.nn.functional.normalize(recon_dists, p=1)
        inputs = torch.nn.functional.normalize(inputs, p=1)  
        DL = torch.mean(torch.sum(-inputs * torch.log(recon_dists + 1e-10), dim=1))

        return DL

    def _train_epoch(self, loader):
        self.model.train()
        train_loss = 0

        for batch_samples in loader:
            batch_dict = batch_samples[0]
            X_bow = batch_dict['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_dict['X_contextual']

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            recon_dists = self.model(X_contextual)

            # backward pass
            dl_loss = self._loss(X_bow, recon_dists)
            loss = dl_loss

            loss.backward()
            self.optimizer.step()

            # compute train loss
            train_loss += loss.item()

        return train_loss

    def fit(self, train_dataset, validation_dataset=None):
        self.train_data = train_dataset
        self.validation_data = validation_dataset
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        train_loss = 0
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            s = datetime.datetime.now()
            train_loss = self._train_epoch(train_loader)
            e = datetime.datetime.now()
            pbar.update(1)
            if self.validation_data is not None:
                validation_loader = DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_data_loader_workers)
                s = datetime.datetime.now()
                val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                pbar.set_description("Epoch: [{}/{}]\t Train Loss: {}\tValid Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, train_loss, val_loss, e - s))

            pbar.set_description("Epoch: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch + 1, self.num_epochs, train_loss, e - s))
            
            if (epoch + 1) % 10 == 0:
                val_res = self._predict(validation_loader)
                record = open('./'+self.config['version']+'_'+self.config['model']+'_'+self.config['dataset']+'_'+self.config['target']+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print('EPOCH', epoch + 1)
                record.write('EPOCH '+ str(epoch + 1) + '\n')
                for key,val in val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")

        pbar.close()

    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                batch_dict = batch_samples[0]
                X_bow = batch_dict['X_bow']
                X_bow = X_bow.reshape(X_bow.shape[0], -1)
                X_contextual = batch_dict['X_contextual']

                if self.USE_CUDA:
                    X_bow = X_bow.cuda()
                    X_contextual = X_contextual.cuda()

                # forward pass
                self.model.zero_grad()
                recon_dists = self.model(X_contextual)
                
                dl_loss = self._loss(X_bow, recon_dists)

                loss = dl_loss


                # compute train loss
                val_loss += loss.item()

        return val_loss

    def _predict(self, loader):
        self.model.eval()
        results = defaultdict(list)
        for batch_samples in loader:
            # batch_size x vocab_size
            batch_dict = batch_samples[0]
            X_bow = batch_dict['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_dict['X_contextual']

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            recon_dists = self.model(X_contextual)
            
            dl_loss = self._loss(X_bow, recon_dists)
            
            # Semantic Prcision for reconstruct
            precision_scores, word_result = semantic_precision_all(recon_dists, X_bow, self.word_embeddings, self.tp_vocab, k=self.config['topk'], th = self.config['threshold'])
            for k, v in precision_scores.items():
                results['[Recon] Semantic Precision@{}'.format(k)].append(v)
                
            # Precision for reconstruct
            precision_scores = retrieval_precision_all(recon_dists, X_bow, k=self.config['topk'])
            for k, v in precision_scores.items():
                results['[Recon] Precision@{}'.format(k)].append(v)

            # NDCG for reconstruct
            ndcg_scores = retrieval_normalized_dcg_all(recon_dists, X_bow, k=self.config['topk'])
            for k, v in ndcg_scores.items():
                results['[Recon] ndcg@{}'.format(k)].append(v)
        
        for k in results:
            results[k] = np.mean(results[k])

        return results

    def get_reconstruct(self, testing_set):
        """Testing epoch."""
        self.test_data = testing_set
        loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        self.model.eval()
        recon_lists = []
        bow_lists = []
        documents_lists = ()
        for batch_samples in loader:
            # batch_size x vocab_size
            batch_dict = batch_samples[0]
            X_bow = batch_dict['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_dict['X_contextual']
            X_documents = batch_samples[1]

            # label list
            bow_lists.append(X_bow)

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            recon_dists = self.model(X_contextual)

            dl_loss = self._loss(X_bow, recon_dists)

            # recontruct list
            recon_lists.append(recon_dists)

            # raw documents list
            documents_lists = documents_lists + X_documents

        return torch.cat(recon_lists, dim=0).cpu().detach().numpy(), torch.cat(bow_lists, dim=0).cpu().detach().numpy(), documents_lists


class MLPDecoder(MLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

