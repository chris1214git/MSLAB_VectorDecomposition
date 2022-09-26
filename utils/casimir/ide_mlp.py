import os
import sys
import math
import nltk
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import argparse
import numpy as np
import multiprocessing as mp
from collections import defaultdict
import multiprocessing as mp

sys.path.append("../")
from load_pretrain_label import load_preprocess_document_labels
from model.ide_gan_decoder import IDEDataset
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs, merge_targets
from torch.utils.data import Dataset, DataLoader
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(15)

def Singular_MythNet(y_pred, y_true, eps=1e-10):
	# ListNet switch softmax to L1 norm
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded. 
    #     ex: same as above.
    # (3) eps: a small number to avoid error when computing log operation. 
    #     ex: log0 will cause error while log(0+eps) will not.

    y_pred = torch.sigmoid(y_pred) 
    y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=1)
    # y_true = torch.nn.functional.softmax(y_true, dim=1) 
    y_true = torch.nn.functional.normalize(y_true, dim=1, p=1)
    pred = y_pred + eps
    pred_log = torch.log(pred)

    return torch.sum(-y_true * pred_log, dim=1)

def generate_dataset(config, balance=False):    
    # Data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]
    print('[INFO] Load corpus done.')

    # Generating document embedding
    while True:
        try:
            doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])
            break
        except:
            print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)
    print('[INFO] Generate embedding done.')
    
    # Generate Decode target & Vocabulary
    if config['target'] == 'keybert' or config['target'] == 'yake':
        labels, vocabularys= load_preprocess_document_labels(config)
        label = labels[config['target']].toarray()
    else:
        labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
        label = labels[config['target']]
        vocabularys = vocabularys[config['target']]
    print('[INFO] Load label done.')
    
    # generate idx to token
    id2token = {k: v for k, v in zip(range(0, len(vocabularys)), vocabularys)}
    print('[INFO] Generate id2token done.')
    
    idx = np.arange(len(unpreprocessed_corpus))
    np.random.shuffle(idx)
    train_length = int(len(unpreprocessed_corpus) * 0.8)
    train_idx = idx[:train_length]
    valid_idx = idx[train_length:]

    train_unpreprocessed_corpus = list(np.array(unpreprocessed_corpus)[train_idx])
    valid_unpreprocessed_corpus = list(np.array(unpreprocessed_corpus)[valid_idx])
    train_embs = np.array(doc_embs)[train_idx]
    valid_embs = np.array(doc_embs)[valid_idx]
    train_label = np.array(label)[train_idx]
    valid_label = np.array(label)[valid_idx]
    
    # Generate labeled mask
    label_masks = np.zeros((train_embs.shape[0], 1), dtype=bool)
    num_labeled_data = int(train_embs.shape[0] * config['ratio'])
    while True:
        if num_labeled_data > 0:
            idx = random.randrange(0, train_embs.shape[0])
            if label_masks[idx] == 0:
                label_masks[idx] = 1
                num_labeled_data -= 1
        else:
            break
    print('[INFO] mask labels done.')

    # Balance data if required
    original_num_data = train_embs.shape[0]
    if config['ratio'] != 1 and balance:
        for idx in range(original_num_data): 
            if label_masks[idx]:
                balance = int(1/config['ratio'])
                balance = int(math.log(balance,2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    train_unpreprocessed_corpus.append(train_unpreprocessed_corpus[idx])
                    train_embs = np.concatenate((train_embs, train_embs[idx].reshape(1, train_embs.shape[1])), axis=0)
                    train_label = np.concatenate((train_label, train_label[idx].reshape(1, train_label.shape[1])), axis=0)
                    label_masks = np.concatenate((label_masks, label_masks[idx].reshape(1, label_masks.shape[1])), axis=0)
        print('[INFO] balance done.')
    
    training_set = IDEDataset(train_unpreprocessed_corpus, train_embs, train_label, label_masks)
    validation_set = IDEDataset(valid_unpreprocessed_corpus, valid_embs, valid_label, np.ones((valid_embs.shape[0], 1), dtype=bool))
    
    return training_set, validation_set, vocabularys, id2token, device 

class DecoderNetwork(nn.Module):
    def __init__(self, input_dim=768, output_dim=100, num_labels=2, dropout=0.2):
        super(DecoderNetwork, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        # self.logit = nn.Linear(output_dim, num_labels)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, embs):
        recons = self.decoder(embs)
        # logits = self.logit(recons)
        # probs = self.softmax(logits)
        return recons# , logits, probs
class MLPDecoder:
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

        self.decoder = DecoderNetwork(input_dim=contextual_dim, output_dim=len(vocab), num_labels=2, dropout=dropout)
        
        if config['optim'] == 'AdamW':
            self.dis_optimizer = AdamW(self.decoder.parameters(), lr=config['lr'], eps=eps)
        else:
            self.dis_optimizer = Adam(self.decoder.parameters(), lr=config['lr'], betas=(self.momentum, 0.99), weight_decay=config['weight_decay'])

        if config['scheduler']:
            num_training_steps = int(len(train_set) / config['batch_size'] * config['epochs'])
            num_warmup_steps = int(num_training_steps * config['warmup_proportion'])
            self.dis_optimizer = AdamW(self.decoder.parameters(), lr=config['lr'], eps=eps)
            if config['warmup'] == 'linear':
                self.dis_scheduler = get_linear_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
            else:
                self.dis_scheduler = get_constant_schedule_with_warmup(self.dis_optimizer, num_warmup_steps=num_warmup_steps)

    def training(self, epoch, loader):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config['epochs']))
        print('Training...')

        dis_train_loss = 0

        self.decoder.train()
        
        total_sample = 0
        for batch, (corpus, embs, labels, masks) in enumerate(loader):
            real_embs, labels, masks = embs.to(self.device), labels.to(self.device), masks.to(self.device)

            recons = self.decoder(real_embs)
    
            # Decoder's LOSS
            recon_loss = torch.masked_select(Singular_MythNet(recons, labels), torch.flatten(masks))
            labeled_count = recon_loss.type(torch.float32).numel()
            total_sample += labeled_count
            if labeled_count == 0:
                continue
            else:
                D_L_Supervised = torch.mean(recon_loss)         
            dis_loss = D_L_Supervised

            self.dis_optimizer.zero_grad()

            dis_loss.backward() 
            
            self.dis_optimizer.step()

            if config['scheduler']:
                self.dis_scheduler.step()

            dis_train_loss += dis_loss.item()
        avg_dis_train_loss = dis_train_loss / len(loader)             

        print("")
        print("  Average training loss decoder: {0:.3f}".format(avg_dis_train_loss))
        print("  Sample num: {}".format(total_sample))
        return avg_dis_train_loss

    def validation(self, loader):
        self.decoder.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (corpus, embs, labels, masks) in enumerate(loader):
                embs, labels = embs.to(self.device), labels.to(self.device)
                recons = self.decoder(embs)
                
                # Precision for reconstruct
                precision_scores = retrieval_precision_all(recons, labels, k=config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v1@{}'.format(k)].append(v)
                
                precision_scores = retrieval_precision_all_v2(recons, labels, k=config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v2@{}'.format(k)].append(v)

                # NDCG for reconstruct
                ndcg_scores = retrieval_normalized_dcg_all(recons, labels, k=config['topk'])
                for k, v in ndcg_scores.items():
                    results['[Recon] ndcg@{}'.format(k)].append(v)

        for k in results:
            results[k] = np.mean(results[k])
                
        return results

    def fit(self):
        self.decoder.to(self.device)

        train_loader = DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.num_data_loader_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.num_data_loader_workers)

        dis_train_loss = 0

        for epoch in range(self.config['epochs']):
            dis_train_loss = self.training(epoch, train_loader)
            if (epoch + 1) % 10 == 0:
                val_res = self.validation(valid_loader)
                record = open('./ide_mlp_'+self.config['experiment']+'_'+self.config['dataset']+str(int(config['ratio']*100))+'_'+self.config['encoder']+'_'+self.config['target']+'_loss_'+self.config['loss']+'_lr'+str(self.config['lr'])+'_optim'+self.config['optim']+'_batch'+str(self.config['batch_size'])+'_weightdecay'+str(self.config['weight_decay'])+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                for key,val in val_res.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--experiment', type=str, default="check")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--architecture', type=str, default="concatenate")
    parser.add_argument('--activation', type=str, default="sigmoid")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--dataset2', type=str, default=None)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--target', type=str, default='tf-idf-gensim')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='listnet')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--warmup', type=str, default='linear')
    parser.add_argument('--warmup_proportion', type=str, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--check_document', type=bool, default=True)
    args = parser.parse_args()
    
    config = vars(args)
    same_seeds(config["seed"])

    # Parameter
    if config['dataset'] == '20news':
        config['min_df'], config['max_df'], config['min_doc_word'] = 62, 1.0, 15
    elif config['dataset'] == 'agnews':
        config['min_df'], config['max_df'], config['min_doc_word'] = 425, 1.0, 15
    elif config['dataset'] == 'IMDB':
        config['min_df'], config['max_df'], config['min_doc_word'] = 166, 1.0, 15
    elif config['dataset'] == 'wiki':
        config['min_df'], config['max_df'], config['min_doc_word'] = 2872, 1.0, 15
    elif config['dataset'] == 'tweet':
        config['min_df'], config['max_df'], config['min_doc_word'] = 5, 1.0, 15
    
    # Generate dataset
    training_set, validation_set, vocabularys, id2token, device = generate_dataset(config, balance=config['balance'])

    model = MLPDecoder(config, training_set, validation_set, vocabularys, id2token, device)
    model.fit()