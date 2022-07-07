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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
# from tqdm.auto import tqdm

sys.path.append("./")
from utils.loss import Singular_MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings

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
    def __init__(self, input_dim=100, output_dim=768, hidden_dim=512, dropout=0.2):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, noise):
        return self.layers(noise)

class Discriminator(nn.Module):
    def __init__(self, input_dim=768, output_dim=100, num_labels=2, dropout=0.2):
        super(Discriminator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.logit = nn.Linear(output_dim, num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embs):
        recons = self.decoder(embs)
        logits = self.logit(recons)
        probs = self.softmax(logits)
        return recons, logits, probs

class IDEGanDecoder:
    def __init__(self, config, train_set, valid_set, vocab = None, id2token=None, device=None, contextual_dim=768, noise_dim=100, word_embeddings=None, dropout=0.2, momentum=0.99, num_data_loader_workers=mp.cpu_count(), loss_weights=None, eps=1e-8):
        self.config = config
        self.train_set = train_set
        self.valid_set = valid_set
        self.vocab = vocab
        self.id2token = id2token
        self.device = device
        self.contextual_dim = contextual_dim
        self.noise_dim = noise_dim
        self.word_embeddings = word_embeddings
        self.dropout = dropout
        self.momentum = momentum
        self.num_data_loader_workers = num_data_loader_workers
        self.loss_weights = loss_weights
        self.eps = eps
        self.relu = torch.nn.ReLU()
        self.cls_loss = torch.nn.CrossEntropyLoss()

        self.generator = Generator(input_dim=noise_dim, output_dim=contextual_dim, dropout=dropout)
        self.discriminator = Discriminator(input_dim=contextual_dim, output_dim=len(vocab), num_labels=2, dropout=dropout)
        
        if config['optim'] == 'AdamW':
            self.gen_optimizer = AdamW(self.generator.parameters(), lr=config['lr'], eps=eps)
            self.dis_optimizer = AdamW(self.discriminator.parameters(), lr=config['lr'], eps=eps)
        else:
            self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])
            self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])

        if config['scheduler']:
            num_training_steps = int(len(train_set) / config['batch_size'] * config['epochs'])
            num_warmup_steps = int(num_training_steps * config['warmup_proportion'])
            self.gen_optimizer = AdamW(self.generator.parameters(), lr=config['lr'], eps=eps)
            self.dis_optimizer = AdamW(self.discriminator.parameters(), lr=config['lr'], eps=eps)
            if config['warmup'] == 'linear':
                self.gen_scheduler = get_linear_schedule_with_warmup(self.gen_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                self.dis_scheduler = get_linear_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            else:
                self.gen_scheduler = get_constant_schedule_with_warmup(self.gen_optimizer, num_warmup_steps=num_warmup_steps)
                self.dis_scheduler = get_constant_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps)

    def training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.config['epochs']))
        print('Training...')

        gen_train_loss, dis_train_loss = 0, 0

        self.generator.train()
        self.discriminator.train()

        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            real_embs, labels, masks = embs.to(self.device), labels.to(self.device), masks.to(self.device)
            cur_batch_size = embs.shape[0]

            noise = torch.zeros(cur_batch_size, 100, device=self.device).uniform_(0, 1)
            fake_embs = self.generator(noise)

            mixed_embs = torch.cat((real_embs, fake_embs), dim=0)
            recons, logits, probs = self.discriminator(mixed_embs)

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
            # g_emb_reg = torch.mean(torch.pow(torch.mean(D_real_recons, dim=0) - torch.mean(D_fake_recons, dim=0), 2))
            # g_emb_reg = -1 * torch.log(self.relu(torch.nn.functional.cosine_similarity(torch.mean(real_embs, dim=0), torch.mean(fake_embs, dim=0), dim=0)) + self.eps)
            g_emb_reg = torch.reciprocal(torch.abs(torch.nn.functional.cosine_similarity(torch.mean(real_embs, dim=0), torch.mean(fake_embs, dim=0), dim=0)))
            gen_loss = g_loss_d + g_emb_reg
    
            # Disciminator's LOSS
            recon_loss = torch.masked_select(Singular_MythNet(D_real_recons, labels), torch.flatten(masks))
            labeled_count = recon_loss.type(torch.float32).numel()
            
            if labeled_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.mean(recon_loss)         
            D_L_unsupervised1U = self.cls_loss(D_real_logits, torch.ones(cur_batch_size, dtype=torch.long).to(self.device))#-1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.eps))
            D_L_unsupervised2U = self.cls_loss(D_fake_logits, torch.zeros(cur_batch_size, dtype=torch.long).to(self.device))#-1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.eps))
            dis_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

            gen_loss.backward(retain_graph=True)
            dis_loss.backward() 
            
            self.gen_optimizer.step()
            self.dis_optimizer.step()

            if self.config['scheduler']:
                self.gen_scheduler.step()
                self.dis_scheduler.step()

            gen_train_loss += gen_loss.item()
            dis_train_loss += dis_loss.item()

        avg_gen_train_loss = gen_train_loss / len(loader)
        avg_dis_train_loss = dis_train_loss / len(loader)             

        print("")
        print("  Average training loss generetor: {0:.3f}".format(avg_gen_train_loss))
        print("  Average training loss discriminator: {0:.3f}".format(avg_dis_train_loss))

        return avg_gen_train_loss, avg_dis_train_loss

    def validation(self, loader):
        self.generator.eval()
        self.discriminator.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (corpus, embs, labels, masks) in enumerate(loader):
                embs, labels = embs.to(self.device), labels.to(self.device)
                recons, logits, probs = self.discriminator(embs)
                
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

    def fit(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)

        gen_train_loss, dis_train_loss = 0, 0

        for epoch in range(self.config['epochs']):
            gen_train_loss, dis_train_loss = self.training(epoch, train_loader)
            if (epoch + 1) % 10 == 0:
                val_res = self.validation(valid_loader)
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
                print("Discriminator training loss: {0:.3f}".format(dis_train_loss))
                record.write("Discriminator training loss: {0:.3f}\n".format(dis_train_loss))