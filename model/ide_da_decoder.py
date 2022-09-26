from base64 import encode
import sys
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
# from tqdm.auto import tqdm

sys.path.append("./")
from utils.loss import MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings
from model.inference_network import ContextualInferenceNetwork

class IDEDataset(Dataset):
    def __init__(self, docs, corpus, emb, target, real):
        
        assert len(emb) == len(target)
        self.docs = docs
        self.corpus = corpus
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)
        self.real = torch.LongTensor(real)
        
    def __getitem__(self, idx):
        return self.docs[idx], self.corpus[idx], self.emb[idx], self.target[idx], self.real[idx]

    def __len__(self):
        return len(self.emb)

class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

    def forward(self, documents):
        return self.get_docvec(documents)

    def get_docvec(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to(self.device)
        embedding = self.model.bert(**inputs).last_hidden_state[:, 0, :]
        return embedding

class Decoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=100, dropout=0.2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, embs):
        recons = self.decoder(embs)
        return recons

class Classifier(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(Classifier, self).__init__()
        self.logit = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embs):
        logits = self.logit(embs)
        probs = self.softmax(logits)
        return logits, probs

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim=1024, dropout=0.2):
        super(FeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, 1, stride=2),
            nn.ReLU(),
        )
        self.fc1d = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # h_output = (h_input - h_kernel + 2 * padding) / stride + 1
        # w_output = (w_input - w_kernel + 2 * padding) / stride + 1

        ## structure 2
        # self.cnn2d = nn.Sequential(
        #     nn.Conv2d(1, 6, (3, 3), stride=1),
        #     nn.ReLU(),
        # )
        # self.fc2d = nn.Sequential(
        #     nn.Linear(6 * 1 * 766, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, feature_dim)
        # )

        ## strucure 3
        # self.cnn2d = nn.Sequential(
        #     nn.Conv2d(1, 12, (2, 2), stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 1), stride=1)
        # )
        # self.fc2d = nn.Sequential(
        #     nn.Linear(12 * 1 * 767, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, feature_dim)
        # )

        ## strucure 5
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 8, (2, 1), stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 768))
        )
        self.fc2d = nn.Sequential(
            nn.Linear(8 * 2 * 768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.fc_1st = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),
            nn.Linear(input_dim, input_dim),
        )
        self.fc_2nd = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid(),
            nn.Linear(input_dim, input_dim),
        )

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.attention = nn.MultiheadAttention(input_dim, 8)
        self.fcattn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, embs, reals):
        # 1D Convolution
        # conv_embs = self.cnn1d(embs.unsqueeze(dim=-1))
        # feature = self.fc1d(conv_embs.squeeze(dim=-1))

        # 2D Convolution
        embs_1st = self.fc_1st(embs)
        embs_2nd = self.fc_2nd(embs_1st)
        embs_2d = torch.stack((embs, embs_1st, embs_2nd), dim=1)
        conv_embs = self.cnn2d(embs_2d.unsqueeze(dim=1))
        feature = self.fc2d(torch.flatten(conv_embs, 1))

        # attention block
        # q = self.query(embs)
        # k = self.key(embs)
        # v = embs
        # attn_embs, _ = self.attention(q, k, v)
        # feature = self.fcattn(attn_embs)

        feature = reals * embs + (1 - reals) * feature

        return feature

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

        self.decoder = nn.Sequential(
            nn.Linear(contextual_size+vocab_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
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
        emb_word_dist = torch.cat((word_dist_for_decoder, emb), dim=1)
        decoded_word_dist = self.decoder(emb_word_dist)
        recon_dist = decoded_word_dist
        
        return self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, recon_dist
    
    def get_theta(self, target, emb, labels=None):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net(target, emb, labels)
            theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta

class IDEDADecoder:
    def __init__(self, config, label_set, unlabel_set, valid_set, vocab = None, id2token=None, device=None, contextual_dim=768, encoded_dim=768, noise_dim=100, word_embeddings=None, dropout=0.2, momentum=0.99, num_data_loader_workers=mp.cpu_count(), loss_weights=None, eps=1e-8):
        self.config = config
        self.label_set = label_set
        self.unlabel_set = unlabel_set
        self.valid_set = valid_set
        self.merge_set = None
        self.vocab = vocab
        self.id2token = id2token
        self.device = device
        self.contextual_dim = contextual_dim
        self.encoded_dim = encoded_dim
        self.n_components = 50
        self.word_embeddings = word_embeddings
        self.dropout = dropout
        self.momentum = momentum
        self.num_data_loader_workers = num_data_loader_workers
        self.loss_weights = loss_weights
        self.eps = eps
        self.relu = torch.nn.ReLU()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        # model
        self.encoder = Encoder(device)
        if config['model'] == 'tsdm':
            self.decoder = DecoderNetwork(
                    config, device, len(vocab), contextual_dim, word_embeddings, n_components=50, hidden_sizes=(100, 100), activation='relu',
                    dropout=dropout, learn_priors=True)
        else:
            self.decoder = Decoder(input_dim=contextual_dim, output_dim=len(vocab))
        self.classifier = Classifier(input_dim=encoded_dim, output_dim=2)
        self.extractor = FeatureExtractor(input_dim=contextual_dim, feature_dim=encoded_dim)
        
        # optimizer
        if config['optim'] == 'AdamW':
            self.en_optimizer = AdamW(self.encoder.parameters(), lr=config['lr'], eps=eps)
            self.de_optimizer = AdamW(self.decoder.parameters(), lr=config['lr'], eps=eps)
            self.cls_optimizer = AdamW(self.classifier.parameters(), lr=config['lr'], eps=eps)
            self.ex_optimizer = AdamW(self.extractor.parameters(), lr=config['lr'], eps=eps)
        else:
            self.en_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.de_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.ex_optimizer = torch.optim.Adam(self.extractor.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
                   
        # scheduler
        if config['scheduler']:
            num_en_training_steps = int(len(label_set) / config['batch_size'] * config['en_epochs'])   
            num_de_training_steps = int((len(label_set) + len(unlabel_set)) / config['batch_size'] * config['de_epochs'])
            num_en_warmup_steps = int(num_en_training_steps * config['warmup_proportion'])
            num_de_warmup_steps = int(num_de_training_steps * config['warmup_proportion'])    
            if config['warmup'] == 'linear':
                self.en_scheduler = get_linear_schedule_with_warmup(self.en_optimizer, num_warmup_steps=num_en_warmup_steps, num_training_steps=num_en_training_steps)
                self.de_scheduler = get_linear_schedule_with_warmup(self.de_optimizer, num_warmup_steps=num_de_warmup_steps, num_training_steps=num_de_training_steps)
                self.cls_scheduler = get_linear_schedule_with_warmup(self.cls_optimizer, num_warmup_steps=num_de_warmup_steps, num_training_steps=num_de_training_steps)
                self.ex_scheduler = get_linear_schedule_with_warmup(self.ex_optimizer, num_warmup_steps=num_de_warmup_steps, num_training_steps=num_de_training_steps)
            else:
                self.en_scheduler = get_constant_schedule_with_warmup(self.en_optimizer, num_warmup_steps=num_en_warmup_steps)
                self.de_scheduler = get_constant_schedule_with_warmup(self.de_optimizer, num_warmup_steps=num_de_warmup_steps)
                self.cls_scheduler = get_constant_schedule_with_warmup(self.cls_optimizer, num_warmup_steps=num_de_warmup_steps)
                self.ex_scheduler = get_constant_schedule_with_warmup(self.ex_optimizer, num_warmup_steps=num_de_warmup_steps)
                
    def en_training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['en_epochs']))
        print('Encoder Training...')
        
        en_train_loss = 0
        en_train_dis = 0
        en_train_cos = 0
        
        self.encoder.train()
        self.extractor.train()

        for batch, (docs, corpus, embs, labels, reals) in enumerate(loader):
            real_embs, reals = embs.to(self.device), reals.to(self.device)
            
            real_embs_t = real_embs
            # real_embs_t = self.extractor(real_embs, reals)
            
            # fake label from BERT            
            fake_embs = self.encoder(corpus).to(self.device)
            fake_embs = self.extractor(fake_embs, torch.zeros([embs.shape[0], 1], dtype=torch.long).to(self.device))

            # Encoder's LOSS
            # e_cos = torch.nn.functional.cosine_similarity(torch.mean(real_embs_t, dim=0), torch.mean(fake_embs, dim=0), dim=0)
            e_cos = torch.mean(torch.nn.functional.cosine_similarity(real_embs_t, fake_embs))
            # e_feat_emb = torch.mean(torch.pow(torch.mean(real_embs_t, dim=0) - torch.mean(fake_embs, dim=0), 2))
            # e_feat_emb =  ContrastiveLoss(real_embs_t, fake_embs, torch.eye(real_embs_t.shape[0], requires_grad=True).to(self.device))
            e_feat_emb = torch.mean(torch.mean(torch.cdist(fake_embs, real_embs_t, p=2), dim=0), dim=0).squeeze()
            # e_feat_emb = self.kl_loss(fake_embs, real_embs_t)
            # e_feat_emb = self.mse_loss(fake_embs, real_embs_t)
            # en_loss = e_feat_emb + (1 - e_cos)
            en_loss = e_feat_emb + embs.shape[0] * (1 - e_cos)
            

            self.en_optimizer.zero_grad()
            self.ex_optimizer.zero_grad()
            en_loss.backward()
            self.en_optimizer.step()
            self.ex_optimizer.step()
            if self.config['scheduler']:
                self.en_scheduler.step()
                self.ex_scheduler.step()

            en_train_loss += en_loss.item()
            en_train_dis += e_feat_emb
            en_train_cos += e_cos

        avg_en_train_loss = en_train_loss / len(loader)        
        avg_en_train_dis = en_train_dis / len(loader) 
        avg_en_train_cos = en_train_cos / len(loader) 

        return avg_en_train_loss, avg_en_train_dis, avg_en_train_cos

    def en_validation(self, loader):
        
        en_val_loss = 0
        en_val_dis = 0
        en_val_cos = 0
        
        self.encoder.eval()
        self.extractor.eval()
        
        with torch.no_grad():
            for batch, (docs, corpus, embs, labels, reals) in enumerate(loader):
                embs, reals = embs.to(self.device), reals.to(self.device)

                real_embs_t = embs
                # real_embs_t = self.extractor(embs, reals)
                
                fake_embs = self.encoder(corpus).to(self.device)
                fake_embs = self.extractor(fake_embs, torch.zeros([embs.shape[0], 1], dtype=torch.long).to(self.device))

                # e_cos = torch.nn.functional.cosine_similarity(torch.mean(real_embs_t, dim=0), torch.mean(fake_embs, dim=0), dim=0)
                e_cos = torch.mean(torch.nn.functional.cosine_similarity(real_embs_t, fake_embs))
                # e_feat_emb = torch.mean(torch.pow(torch.mean(real_embs_t, dim=0) - torch.mean(fake_embs, dim=0), 2))
                # e_feat_emb =  ContrastiveLoss(real_embs_t, fake_embs, torch.eye(real_embs_t.shape[0], requires_grad=True).to(self.device))
                e_feat_emb = torch.mean(torch.mean(torch.cdist(fake_embs, real_embs_t, p=2), dim=0), dim=0).squeeze()
                # e_feat_emb = self.kl_loss(fake_embs, real_embs_t)
                # e_feat_emb = self.mse_loss(fake_embs, real_embs_t)
                # en_loss = e_feat_emb + (1 - e_cos)
                en_loss = e_feat_emb + embs.shape[0] * (1 - e_cos)
                
                en_val_loss += en_loss
                en_val_dis += e_feat_emb
                en_val_cos += e_cos
               
            avg_en_val_loss = en_val_loss / len(loader)
            avg_en_val_dis = en_val_dis / len(loader)
            avg_en_val_cos = en_val_cos / len(loader)
        
        return avg_en_val_loss, avg_en_val_dis, avg_en_val_cos
        
    def generate_fake_data(self, loader):
        self.encoder.eval()
        doc_list = []
        corpus_list = []
        emb_list = []
        label_list = []

        for batch, (doc, corpus, embs, labels, reals) in enumerate(loader):
            fake_embs = self.encoder(corpus).detach().cpu()
            for raw_doc, pro_doc, emb, label in zip(doc, corpus, fake_embs, labels):
                doc_list.append(raw_doc)
                corpus_list.append(pro_doc)
                emb_list.append(emb.numpy())
                label_list.append(label.numpy())

        return IDEDataset(doc_list, corpus_list, np.array(emb_list), np.array(label_list), np.zeros((len(emb_list), 1)))

    def en_fit(self):
        self.encoder.to(self.device)
        self.extractor.to(self.device)

        label_loader = DataLoader(self.label_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        unlabel_loader = DataLoader(self.unlabel_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)

        for epoch in range(self.config['en_epochs']):
            en_train_loss, en_train_dis, en_train_cos = self.en_training(epoch, label_loader)
            
            en_val_loss, en_val_dis, en_val_cos = self.en_validation(unlabel_loader)
            
            print('---------------------------------------')
            print("Training: ")
            print(" [Encoder] Average training loss: {0:.3f}".format(en_train_loss))
            print(" [Encoder] Average training distance: {0:.3f}".format(en_train_dis))
            print(" [Encoder] Average training cosine similarity: {0:.3f}".format(en_train_cos))
            print("Validation:")
            print(" [Encoder] Average validation loss: {0:.3f}".format(en_val_loss))
            print(" [Encoder] Average validation dis: {0:.3f}".format(en_val_dis))
            print(" [Encoder] Average validation cosine similarity: {0:.3f}".format(en_val_cos))
            
            withscheduler = 'with_scheduler' if self.config['scheduler'] else '_without_scheduler'
            withbalance = 'with_balance' if self.config['balance'] else '_without_balance'
            record = open('./ide_da_encoder_'+self.config['experiment']+'_'+self.config['dataset']+str(int(self.config['ratio'] * 100))+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+withscheduler+withbalance+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
            record.write('-------------------------------------------------\n')
            record.write("Training:\n")
            record.write(" [Encoder] Average training loss: {0:.3f}\n".format(en_train_loss))
            record.write(" [Encoder] Average training distance: {0:.3f}\n".format(en_train_dis))
            record.write(" [Encoder] Average training cosine similarity: {0:.3f}\n".format(en_train_cos))
            record.write("Validation:\n")
            record.write(" [Encoder] Average validation loss: {0:.3f}\n".format(en_val_loss))
            record.write(" [Encoder] Average validation dis: {0:.3f}\n".format(en_val_dis))
            record.write(" [Encoder] Average validation cosine similarity: {0:.3f}\n".format(en_val_cos))

        self.merge_set = torch.utils.data.ConcatDataset([self.generate_fake_data(unlabel_loader), self.label_set])

    def cls_training(self, epoch, loader):        
        cls_train_loss = 0
        
        self.classifier.train()
        self.decoder.eval()
        self.extractor.eval()

        for batch, (docs, corpus, embs, labels, reals) in enumerate(loader):
            embs, reals = embs.to(self.device), reals.to(self.device)   

            # Extract features
            features = self.extractor(embs, reals)
            # features = embs
            
            # Classifier discrimiate features
            logits, probs = self.classifier(features)
            cls_loss = self.cross_entropy(probs, torch.flatten(reals))
            
            self.cls_optimizer.zero_grad()
            
            cls_loss.backward()
            
            self.cls_optimizer.step()
            if self.config['scheduler']:
                self.cls_scheduler.step()

            cls_train_loss += cls_loss.item()
  
        avg_cls_train_loss = cls_train_loss / len(loader)    

        return avg_cls_train_loss

    def de_training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['de_epochs']))
        print('Decoder Training...')
        
        de_train_loss = 0
        cls_train_loss = 0

        de_loss_weight = 1
        cls_loss_weight = 10
        
        samples_processed = 0
        
        self.classifier.eval()
        self.decoder.train()
        self.extractor.train()

        for batch, (docs, corpus, embs, labels, reals) in enumerate(loader):
            embs, labels, reals = embs.to(self.device), labels.to(self.device), reals.to(self.device)   

            # Extract features
            features = self.extractor(embs, reals)
            # distance_loss = torch.mean(torch.pow(torch.mean(features * reals, dim=0) - torch.mean(features * (1 - reals), dim=0), 2))
            # features = embs
            
            # Classifier discrimiate features
            logits, probs = self.classifier(features)
            # cls_loss = self.cross_entropy(probs, torch.ones([embs.shape[0]], dtype=torch.long).to(self.device)) + self.cross_entropy(probs, torch.zeros([embs.shape[0]], dtype=torch.long).to(self.device))
            cls_loss = self.cross_entropy(probs, torch.flatten(1 - reals))
            # cls_loss = torch.reciprocal(self.cross_entropy(probs, torch.flatten(reals)) + self.eps)

            if self.config['model'] == 'tsdm':
                labels = labels.reshape(labels.shape[0], -1)
                prior_mean, prior_variance, posterior_mean, posterior_variance,\
                    posterior_log_variance, word_dists, recons = self.decoder(features, labels)

                var_division = torch.sum(posterior_variance / prior_variance, dim=1)
                diff_means = prior_mean - posterior_mean
                diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
                logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
                
                samples_processed = labels.size()[0]
                
                KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)
                RL = torch.sum(-labels * torch.log(word_dists + 1e-10), dim=1)
                DL = MythNet(recons, labels)

                loss = (KL + RL) / samples_processed + DL
                loss = loss.sum()

                de_loss = de_loss_weight * loss + cls_loss_weight * cls_loss
                ex_loss = de_loss_weight * DL + cls_loss_weight * cls_loss
            else:
                # Decoder reconstruct embs
                recons = self.decoder(features)

                # ListNet Loss
                de_loss = de_loss_weight * MythNet(recons, labels) + cls_loss_weight * cls_loss  
                ex_loss = de_loss + cls_loss_weight * cls_loss     

            self.de_optimizer.zero_grad()
            self.ex_optimizer.zero_grad()
            de_loss.backward()
            # ex_loss.backward()
            self.de_optimizer.step()
            self.ex_optimizer.step()
            if self.config['scheduler']:
                self.de_scheduler.step()
                self.ex_scheduler.step()

            de_train_loss += de_loss.item()

        avg_de_train_loss = de_train_loss / len(loader)      

        return avg_de_train_loss
    
    def de_validation(self, loader):
        de_val_loss = 0
        cls_val_loss = 0

        de_loss_weight = 1
        cls_loss_weight = 10
        
        self.classifier.eval()
        self.decoder.eval()
        self.extractor.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (docs, corpus, embs, labels, reals) in enumerate(loader):
                embs, labels, reals = embs.to(self.device), labels.to(self.device), reals.to(self.device)
                
                # Extract features
                features = self.extractor(embs, reals)
                # distance_loss = torch.mean(torch.pow(torch.mean(features * reals, dim=0) - torch.mean(features * (1 - reals), dim=0), 2))
                # features = embs

                # Clssifier
                logits, probs = self.classifier(features)
                cls_loss = self.cross_entropy(probs, torch.flatten(reals))
                # de_cls_loss = self.cross_entropy(probs, torch.ones([embs.shape[0]], dtype=torch.long).to(self.device)) + self.cross_entropy(probs, torch.zeros([embs.shape[0]], dtype=torch.long).to(self.device))
                # de_cls_loss = torch.reciprocal(self.cross_entropy(probs, torch.flatten(reals)) + self.eps)
                de_cls_loss = self.cross_entropy(probs, torch.flatten(1 - reals))

                if self.config['model'] == 'tsdm':
                    labels = labels.reshape(labels.shape[0], -1)
                    prior_mean, prior_variance, posterior_mean, posterior_variance,\
                        posterior_log_variance, word_dists, recons = self.decoder(features, labels)

                    var_division = torch.sum(posterior_variance / prior_variance, dim=1)
                    diff_means = prior_mean - posterior_mean
                    diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
                    logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(dim=1)

                    KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)
                    RL = torch.sum(-labels * torch.log(word_dists + 1e-10), dim=1)
                    DL = MythNet(recons, labels)

                    topic_loss = KL + RL + DL
                    topic_loss = topic_loss.sum()

                    de_loss = de_loss_weight * DL
                else:
                    # Decoder reconstruct
                    recons = self.decoder(features)
                    
                    # ListNet Loss
                    de_loss = de_loss_weight * MythNet(recons, labels)# + cls_loss_weight * de_cls_loss# + distance_loss

                de_val_loss += de_loss.item()
                cls_val_loss += cls_loss.item()
                
                # Precision for reconstruct
                precision_scores = retrieval_precision_all(recons, labels, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v1@{}'.format(k)].append(v)
                
                precision_scores = retrieval_precision_all_v2(recons, labels, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v2@{}'.format(k)].append(v)

                # NDCG for reconstruct
                ndcg_scores = retrieval_normalized_dcg_all(recons, labels, k=self.config['topk'])
                for k, v in ndcg_scores.items():
                    results['[Recon] ndcg@{}'.format(k)].append(v)
        
        avg_de_val_loss = de_val_loss / len(loader)
        avg_cls_val_loss = cls_val_loss / len(loader)
        
        for k in results:
            results[k] = np.mean(results[k])
                
        return avg_de_val_loss, results, avg_cls_val_loss
    
    def de_fit(self):
        self.decoder.to(self.device)
        # self.extractor.to(self.device)
        self.classifier.to(self.device)

        fake_weight = [len(self.label_set)] * len(self.unlabel_set)
        real_weight = [len(self.unlabel_set)] * len(self.label_set)

        sampler = WeightedRandomSampler(fake_weight+real_weight, len(self.merge_set), replacement=True)

        train_loader = DataLoader(self.merge_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers, sampler=sampler)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)
        
        for epoch in range(self.config['de_epochs']):
            de_train_loss = self.de_training(epoch, train_loader)
            cls_train_loss = self.cls_training(epoch, train_loader)

            print('---------------------------------------')
            print("Training: ")
            print(" [Decoder] Average training loss: {0:.3f}".format(de_train_loss))
            print(" [Classifier] Average training loss: {0:.3f}".format(cls_train_loss))
            if (epoch + 1) % 10 == 0:
                de_val_loss, de_val_res, cls_val_loss = self.de_validation(valid_loader)
            
                print("Validation:")
                print(" [Decoder] Average validation loss: {0:.3f}".format(de_val_loss))
                print(" [Classifier] Average validation loss: {0:.3f}".format(cls_val_loss))
                print(" [Decoder] Average validation result:")

                withscheduler = 'with_scheduler' if self.config['scheduler'] else '_without_scheduler'
                withbalance = 'with_balance' if self.config['balance'] else '_without_balance'
                record = open('./ide_da_decoder_'+self.config['experiment']+'_'+self.config['dataset']+str(int(self.config['ratio'] * 100))+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+withscheduler+withbalance+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                record.write('-------------------------------------------------\n')
                record.write("Training:\n")
                record.write(" [Decoder] Average training loss: {0:.3f}\n".format(de_train_loss))
                record.write(" [Classifier] Average training loss: {0:.3f}\n".format(cls_train_loss))
                record.write("Validation:\n")
                record.write(" [Decoder] Average validation loss: {0:.3f}\n".format(de_val_loss))
                record.write(" [Classifier] Average validation loss: {0:.3f}\n".format(cls_val_loss))
                record.write(" [Decoder] Average validation result:\n")

                for key,val in de_val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")
    