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
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
# from tqdm.auto import tqdm

sys.path.append("./")
from utils.loss import Singular_MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings
from model.inference_network import ContextualInferenceNetwork

class IDEDataset(Dataset):
    def __init__(self, corpus, emb, target, mask):
        
        assert len(emb) == len(target)
        self.corpus = corpus
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)
        self.mask = torch.BoolTensor(mask)
        
    def __getitem__(self, idx):
        return self.corpus[idx], self.emb[idx], self.target[idx], self.mask[idx]

    def __len__(self):
        return len(self.emb)

class Generator(nn.Module):
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

class Discriminator(nn.Module):
    def __init__(self, input_dim=768, output_dim=100, dropout=0.2):
        super(Discriminator, self).__init__()
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
    
class MLPDecoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, dropout=0.2):
        super(MLPDecoder, self).__init__()
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
    
class VariationalAE(nn.Module):
    def __init__(self, config, device, vocab_size, contextual_size=768, encoded_size=768, n_components=50, hidden_sizes=(100,100), activation='relu', dropout=0.2, learn_priors=True):
        super(VariationalAE, self).__init__()

        assert activation in ['softplus', 'relu']

        self.config = config
        self.device = device
        self.vocab_size = vocab_size
        self.contextual_size = contextual_size
        self.encoded_size = encoded_size
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
            nn.Linear(encoded_size, contextual_size*4),
            nn.BatchNorm1d(contextual_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(contextual_size*4, contextual_size),
            nn.BatchNorm1d(contextual_size),
        )
        
        # topic model architecture
        self.inf_net = ContextualInferenceNetwork(encoded_size, contextual_size, n_components, hidden_sizes, activation, label_size=0)
        
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components).to(device)
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor([topic_prior_variance] * n_components).to(device)
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, encoded_size).to(device)
        self.beta = nn.Parameter(self.beta)
        
        nn.init.xavier_uniform_(self.beta)
        
        self.beta_batchnorm = nn.BatchNorm1d(encoded_size, affine=False)
        
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
        
        self.topic_word_matrix = self.beta
        
        # decode
        recon = self.decoder(word_dist);
        return self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, recon
    
    def get_theta(self, target, emb, labels=None):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net(target, emb, labels)
            theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta


class IDEAEDecoder:
    def __init__(self, config, train_set, valid_set, vocab = None, id2token=None, device=None, contextual_dim=768, encoded_dim=768, noise_dim=100, word_embeddings=None, dropout=0.2, momentum=0.99, num_data_loader_workers=mp.cpu_count(), loss_weights=None, eps=1e-8):
        self.config = config
        self.train_set = train_set
        self.valid_set = valid_set
        self.vocab = vocab
        self.id2token = id2token
        self.device = device
        self.contextual_dim = contextual_dim
        self.encoded_dim = encoded_dim
        self.noise_dim = noise_dim
        self.word_embeddings = word_embeddings
        self.dropout = dropout
        self.momentum = momentum
        self.num_data_loader_workers = num_data_loader_workers
        self.loss_weights = loss_weights
        self.eps = eps
        self.relu = torch.nn.ReLU()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # model
        self.vae = VariationalAE(config, device, len(vocab), contextual_dim, encoded_dim, 50, (100, 100), 'relu', 0.2, True)
        self.decoder = MLPDecoder(encoded_dim, len(vocab), 0.2)
        self.generator = Generator(device)
        self.discriminator = Discriminator(input_dim=contextual_dim, output_dim=len(vocab), dropout=dropout)
        self.classifier = Classifier(input_dim=contextual_dim, output_dim=2)
        
        # optimizer
        if config['optim'] == 'AdamW':
            self.vae_optimizer = AdamW(self.vae.parameters(), lr=config['ae_lr'], eps=eps)
            self.decoder_optimizer = AdamW(self.decoder.parameters(), lr=config['lr'], eps=eps)
            self.gen_optimizer = AdamW(self.generator.parameters(), lr=config['lr'], eps=eps)
            self.dis_optimizer = AdamW(self.discriminator.parameters(), lr=config['lr'], eps=eps)
            self.cls_optimizer = AdamW(self.classifier.parameters(), lr=config['lr'], eps=eps)
        else:
            self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=config['ae_lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
        
        # scheduler
        if config['scheduler']:
            num_training_steps = int(len(train_set) / config['batch_size'] * config['epochs'])
            num_warmup_steps = int(num_training_steps * config['warmup_proportion'])
            self.vae_optimizer = AdamW(self.vae.parameters(), lr=config['ae_lr'], eps=eps)
            self.decoder_optimizer = AdamW(self.decoder.parameters(), lr=config['lr'], eps=eps)
            self.gen_optimizer = AdamW(self.generator.parameters(), lr=config['lr'], eps=eps)
            self.dis_optimizer = AdamW(self.discriminator.parameters(), lr=config['lr'], eps=eps)
            self.cls_optimizer = AdamW(self.classifier.parameters(), lr=config['lr'], eps=eps)
            if config['warmup'] == 'linear':
                self.vae_scheduler = get_linear_schedule_with_warmup(self.vae_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                self.decoder_scheduler = get_linear_schedule_with_warmup(self.decoder_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                self.gen_scheduler = get_linear_schedule_with_warmup(self.gen_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                self.dis_scheduler = get_linear_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                self.cls_scheduler = get_linear_schedule_with_warmup(self.cls_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            else:
                self.vae_scheduler = get_constant_schedule_with_warmup(self.vae_optimizer, num_warmup_steps=num_warmup_steps)
                self.decoder_scheduler = get_constant_schedule_with_warmup(self.decoder_optimizer, num_warmup_steps=num_warmup_steps)
                self.gen_scheduler = get_constant_schedule_with_warmup(self.gen_optimizer, num_warmup_steps=num_warmup_steps)
                self.dis_scheduler = get_constant_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps)
                self.cls_scheduler = get_constant_schedule_with_warmup(self.cls_optimizer, num_warmup_steps=num_warmup_steps)
                
    def ae_training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['ae_epochs']))
        print('AutoEncoder Training...')

        ae_train_loss = 0
        ae_train_cos = 0
        
        self.vae.train()

        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            embs, masks = embs.to(self.device), masks.to(self.device)
            _, _, _, _, _, encoded, decoded = self.vae(embs, embs)
            
            # Loss weight
            cos = torch.nn.functional.cosine_similarity(torch.mean(embs, dim=0), torch.mean(decoded, dim=0), dim=0)
            # w = 1 - cos
            
            # Encode-Decode's Loss
            recon_loss = torch.mean(self.mse_loss(decoded, embs), dim=1)
            mask_loss = torch.masked_select(recon_loss, torch.flatten(~masks))     
            decoded_loss = torch.mean(mask_loss)
            print(decoded_loss)
                       
            self.vae_optimizer.zero_grad()
            decoded_loss.backward() 
            self.vae_optimizer.step()

            ae_train_loss += decoded_loss.item()
            ae_train_cos += cos

        avg_ae_train_loss = ae_train_loss / len(loader)  
        avg_ae_train_cos = ae_train_cos / len(loader)
        

        print("")
        print("  Average training loss AutoEncoder: {0:.3f}".format(avg_ae_train_loss))
        print("  Average training Cosine-Similarity AutoEncoder: {0:.3f}".format(avg_ae_train_cos))

        return avg_ae_train_loss, avg_ae_train_cos
    
    def mlp_training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['epochs']))
        print('Decoder Training...')

        decode_train_loss = 0

        self.vae.eval()
        self.decoder.train()

        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            embs, labels, masks = embs.to(self.device), labels.to(self.device), masks.to(self.device)

            # VAE transform
            _, _, _, _, _, encoded, _ = self.vae(embs, embs)   
            
            # Decode
            recons = self.decoder(encoded)
            
            # Decoder's LOSS
            mask_loss = torch.masked_select(Singular_MythNet(recons, labels), torch.flatten(masks))
            labeled_count = mask_loss.type(torch.float32).numel()
            if labeled_count == 0:
                continue
            else:
                decoded_loss = torch.mean(mask_loss)
            
            self.decoder_optimizer.zero_grad()
            decoded_loss.backward() 
            self.decoder_optimizer.step()

            decode_train_loss += decoded_loss.item()

        avg_decoded_train_loss = decode_train_loss / len(loader)             

        print("")
        print("  Average training loss decoder: {0:.3f}".format(avg_decoded_train_loss))

        return avg_decoded_train_loss
    
    def gen_training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['epochs']))
        print('Bert GAN Training...')
        
        gen_train_loss = 0
        
        #self.vae.train()
        self.generator.train()
        self.discriminator.eval()
        self.classifier.eval()

        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            real_embs, labels, masks = embs.to(self.device), labels.to(self.device), masks.to(self.device)
            cur_batch_size = embs.shape[0]
            
            # vae transform
            #real_embs_t = self.vae(real_embs)
            real_embs_t = real_embs
            
            # fake label from BERT
            noise = torch.empty(cur_batch_size, dtype=torch.long).random_(len(self.train_set))
            noise_docs = []
            noise_labels = torch.FloatTensor([])
            for i in range(cur_batch_size):     
                noise_docs.append(self.train_set[i][0])
                noise_labels = torch.cat((noise_labels, self.train_set[i][2]))
            fake_labels = torch.reshape(noise_labels, (cur_batch_size, len(self.vocab))).to(self.device)
            
            fake_embs = self.generator(noise_docs).to(self.device)

            mixed_embs = torch.cat((real_embs_t, fake_embs), dim=0)
            logits, probs = self.classifier(mixed_embs)
            recons = self.discriminator(mixed_embs)         

            recons_list = torch.split(recons, cur_batch_size)
            D_real_recons = recons_list[0]
            D_fake_recons = recons_list[1]
        
            logits_list = torch.split(logits, cur_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]
            
            probs_list = torch.split(probs, cur_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            # Generator's LOSS
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + self.eps))
            g_feat_emb = torch.mean(torch.pow(torch.mean(real_embs_t, dim=0) - torch.mean(fake_embs, dim=0), 2))
            gen_loss = g_loss_d + g_feat_emb
            

            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()
            if self.config['scheduler']:
                self.gen_scheduler.step()

            gen_train_loss += gen_loss.item()

        avg_gen_train_loss = gen_train_loss / len(loader)           

        print("")
        print("  Average training loss generetor: {0:.3f}".format(avg_gen_train_loss))

        return avg_gen_train_loss
        
    def dis_training(self, epoch, loader):      
        cls_train_loss, dis_train_loss = 0, 0
        
        #self.vae.train()
        self.generator.eval()
        self.discriminator.train()
        self.classifier.train()

        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            real_embs, labels, masks = embs.to(self.device), labels.to(self.device), masks.to(self.device)
            cur_batch_size = embs.shape[0]
            
            # vae transform
            #real_embs_t = self.vae(real_embs)
            real_embs_t = real_embs
            
            # fake label from BERT
            noise = torch.empty(cur_batch_size, dtype=torch.long).random_(len(self.train_set))
            noise_docs = []
            noise_labels = torch.FloatTensor([])
            for i in range(cur_batch_size):     
                noise_docs.append(self.train_set[i][0])
                noise_labels = torch.cat((noise_labels, self.train_set[i][2]))
            fake_labels = torch.reshape(noise_labels, (cur_batch_size, len(self.vocab))).to(self.device)
            
            fake_embs = self.generator(noise_docs).to(self.device)

            mixed_embs = torch.cat((real_embs_t, fake_embs), dim=0)
            logits, probs = self.classifier(mixed_embs)
            recons = self.discriminator(mixed_embs)

            recons_list = torch.split(recons, cur_batch_size)
            D_real_recons = recons_list[0]
            D_fake_recons = recons_list[1]
        
            logits_list = torch.split(logits, cur_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]
            
            probs_list = torch.split(probs, cur_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]
            
            # Classifier's Loss
            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.eps))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.eps))
            #D_L_unsupervised1U = self.cls_loss(D_real_logits, torch.ones(cur_batch_size, dtype=torch.long).to(self.device))
            #D_L_unsupervised2U = self.cls_loss(D_fake_logits, torch.zeros(cur_batch_size, dtype=torch.long).to(self.device)) 
            cls_loss =  D_L_unsupervised1U + D_L_unsupervised2U
            
            # Disciminator's LOSS
            recon_loss = torch.masked_select(Singular_MythNet(D_real_recons, labels), torch.flatten(masks))
            g_recon_weight =  D_fake_probs[:, 0]
            fake_recon_loss = Singular_MythNet(D_fake_recons, fake_labels) * g_recon_weight
            labeled_count = recon_loss.type(torch.float32).numel()
            
            if labeled_count == 0:
                D_L_Supervised = torch.mean(fake_recon_loss)
            else:
                D_L_Supervised = torch.mean(recon_loss) + torch.mean(fake_recon_loss)                    
            dis_loss = D_L_Supervised# + cls_loss
            
            self.cls_optimizer.zero_grad()
            cls_loss.backward(retain_graph=True)
            self.cls_optimizer.step()
            if self.config['scheduler']:
                self.cls_scheduler.step()
                
            self.dis_optimizer.zero_grad()
            dis_loss.backward()
            self.dis_optimizer.step()
            if self.config['scheduler']:
                self.dis_scheduler.step()
            
            cls_train_loss += cls_loss.item()
            dis_train_loss += dis_loss.item()
        
        avg_cls_train_loss = cls_train_loss / len(loader)
        avg_dis_train_loss = dis_train_loss / len(loader)           
        
        print("  Average training loss classifier: {0:.3f}".format(avg_cls_train_loss))
        print("  Average training loss discriminator: {0:.3f}".format(avg_dis_train_loss))

        return avg_cls_train_loss, avg_dis_train_loss
        
    def ae_validation(self, loader):
        ae_val_loss = 0
        ae_val_cos = 0
        
        self.vae.eval()
        
        with torch.no_grad():
            for batch, (corpus, embs, labels, masks) in enumerate(loader):
                embs, masks = embs.to(self.device), masks.to(self.device)
                prior_mean, prior_variance, posterior_mean, posterior_variance,\
                posterior_log_variance, encoded, decoded = self.vae(embs, embs)

                # Loss weight
                cos = torch.nn.functional.cosine_similarity(torch.mean(embs, dim=0), torch.mean(decoded, dim=0), dim=0)
                w = 1 - cos
                
                # Encode-Decode's Loss
                recon_loss = torch.mean(self.mse_loss(decoded, embs), dim=1)    
                decoded_loss = torch.mean(recon_loss) * w

                ae_val_loss += decoded_loss.item()
                ae_val_cos += cos
                
            avg_ae_val_loss = ae_val_loss / len(loader)
            avg_ae_val_cos = ae_val_cos / len(loader)
        
        return avg_ae_val_loss, avg_ae_val_cos
    
    def mlp_validation(self, loader):
        self.vae.eval()
        self.decoder.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (corpus, embs, labels, masks) in enumerate(loader):
                embs, labels = embs.to(self.device), labels.to(self.device)
                
                # VAE transform
                _, _, _, _, _, encoded, _ = self.vae(embs, embs)   

                # Decode
                recons = self.decoder(encoded)
                
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

        for k in results:
            results[k] = np.mean(results[k])
                
        return results
    
    def gan_validation(self, loader):
        self.vae.eval()
        self.generator.eval()
        self.classifier.eval()
        self.discriminator.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (corpus, embs, labels, masks) in enumerate(loader):
                embs, labels = embs.to(self.device), labels.to(self.device)
                #embs_t = self.vae(embs, embs)
                embs_t = embs
                
                logits, probs = self.classifier(embs_t)
                recons = self.discriminator(embs_t)
                
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

        for k in results:
            results[k] = np.mean(results[k])
                
        return results
    
    def ae_fit(self):
        self.vae.to(self.device)

        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)
        ae_loss = 0

        for epoch in range(self.config['ae_epochs']):
            ae_loss, ae_cos = self.ae_training(epoch, train_loader)
            if (epoch + 1) % 10 == 0:
                val_loss, val_cos = self.ae_validation(valid_loader)
                withscheduler = 'with_scheduler' if self.config['scheduler'] else '_without_scheduler'
                withbalance = 'with_balance' if self.config['balance'] else '_without_balance'
                record = open('./ae_'+self.config['experiment']+'_'+self.config['dataset']+str(int(self.config['ratio'] * 100))+'_'+self.config['encoder']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+withscheduler+withbalance+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print("AutoEncoder Validation loss: {0:.3f}".format(val_loss))
                record.write("AutoEncoder training loss: {0:.3f}\n".format(val_loss))
                print("AutoEncoder validation Cosine-Similarity: {0:.3f}".format(val_cos))
                record.write("AutoEncoder validation Cosine-Similarity: {0:.3f}\n".format(val_cos))
    
    def mlp_fit(self):
        self.decoder.to(self.device)

        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)

        decoded_train_loss = 0

        for epoch in range(self.config['epochs']):
            decoded_train_loss = self.mlp_training(epoch, train_loader)
            if (epoch + 1) % 10 == 0:
                val_res = self.mlp_validation(valid_loader)
                withscheduler = 'with_scheduler' if self.config['scheduler'] else '_without_scheduler'
                withbalance = 'with_balance' if self.config['balance'] else '_without_balance'
                record = open('./ide_semi_'+self.config['experiment']+'_'+self.config['dataset']+str(int(self.config['ratio'] * 100))+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+withscheduler+withbalance+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                for key,val in val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")
                print("Decoder training loss: {0:.3f}".format(decoded_train_loss))
                record.write("Decoder training loss: {0:.3f}\n".format(decoded_train_loss))

    def gan_fit(self):
        self.generator.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)

        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)

        gen_train_loss, cls_train_loss, dis_train_loss = 0, 0, 0

        for epoch in range(self.config['epochs']):
            gen_train_loss = self.gen_training(epoch, train_loader)
            cls_train_loss, dis_train_loss = self.dis_training(epoch, train_loader)
            if (epoch + 1) % 10 == 0:
                val_res = self.gan_validation(valid_loader)
                withscheduler = 'with_scheduler' if self.config['scheduler'] else '_without_scheduler'
                withbalance = 'with_balance' if self.config['balance'] else '_without_balance'
                record = open('./ide_gan_'+self.config['experiment']+'_'+self.config['dataset']+str(int(self.config['ratio'] * 100))+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+withscheduler+withbalance+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                for key,val in val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")
                print("Generator training loss: {0:.3f}".format(gen_train_loss))
                record.write("Generator training loss: {0:.3f}\n".format(gen_train_loss))
                print("Classifier training loss: {0:.3f}".format(cls_train_loss))
                record.write("Classifier training loss: {0:.3f}\n".format(cls_train_loss))
                print("Discriminator training loss: {0:.3f}".format(dis_train_loss))
                record.write("Discriminator training loss: {0:.3f}\n".format(dis_train_loss))