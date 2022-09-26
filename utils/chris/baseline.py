#!/usr/bin/env python
# coding: utf-8

# # MLP baseline

# In[1]:


import os
import sys
from collections import defaultdict
import numpy as np 
import pandas as pd
import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, retrieval_precision_all_v2, semantic_precision_all, semantic_precision_all_v2, precision_recall_f1_all
from utils.loss import *
from utils.data_loader import load_document
from utils.toolbox import preprocess_document, get_preprocess_document, get_preprocess_document_embs,                          get_preprocess_document_labels, get_preprocess_document_labels_v2, get_word_embs,                          get_free_gpu, merge_targets


# ## Load Data, Label
# label -> bow, tf-idf, keybert, classification

# In[2]:
import argparse
parser = argparse.ArgumentParser(description='dnn decoder baseline')
parser.add_argument('--dataset', type=str, default="20news")
parser.add_argument('--dataset2', type=str, default=None)
parser.add_argument('--model_name', type=str, default='average')
parser.add_argument('--label_type', type=str, default='bow')
parser.add_argument('--eval_f1', action="store_true")
parser.add_argument('--criterion', type=str, default='BCE')
parser.add_argument('--n_gram', type=int, default=1)
parser.add_argument('--n_time', type=int, default=5)
parser.add_argument('--train_size', type=float, default=0.8)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--valid_epoch', type=int, default=10)
parser.add_argument('--h_dim', type=int, default=300)
parser.add_argument('--target_normalization', action="store_true")
parser.add_argument('--save_dir', type=str, default='default')
parser.add_argument('--preprocess_config_dir', type=str, default='parameters_baseline2')
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()
config = vars(args)

dataset = config['dataset']
# cross domain
dataset2 = config['dataset2']
model_name = config['model_name']
label_type = config['label_type']
# 用binary(f1) evaluation或rank evaluation
eval_f1 = config['eval_f1']
criterion = config['criterion']
# 選preprocess config
preprocess_config_dir = config['preprocess_config_dir']
n_gram =  config['n_gram']

# 訓練幾次
n_time = config['n_time']
seed = config['seed']

lr = config['lr']
n_epoch = config['n_epoch']
valid_epoch = config['valid_epoch']
h_dim = config['h_dim']
target_normalization = config['target_normalization']

if dataset2:
    experiment_dir = f'cross_{dataset}_{dataset2}_{model_name}_{label_type}_{criterion}'
else:
    experiment_dir = f'{dataset}_{model_name}_{label_type}_{criterion}'
    
save_dir = os.path.join('experiment', experiment_dir, config['save_dir'])
os.makedirs(save_dir, exist_ok=False)


# In[3]:


def load_training_data(config, dataset):
    preprocess_config_dir = config['preprocess_config_dir']
    with open(os.path.join(f'../chris/{preprocess_config_dir}', f'preprocess_config_{dataset}.json'), 'r') as f:
        preprocess_config = json.load(f)
        
    # load preprocess dataset
    unpreprocessed_docs, preprocessed_docs = get_preprocess_document(**preprocess_config)
    print('doc num', len(preprocessed_docs))

    # get document embeddings
    doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_docs, model_name)
    print('doc_embs', doc_embs.shape)
    
    # load labels
    labels, vocabularys = get_preprocess_document_labels_v2(preprocessed_docs, preprocess_config, preprocess_config_dir, config['n_gram'])    
    # check nonzero numbers
    for k in labels:
        print(k, np.sum(labels[k]!=0), labels[k].shape)
    print(len(vocabularys))
    # select label type
    targets = labels[config['label_type']].toarray()
    vocabularys = vocabularys
    
    return unpreprocessed_docs ,preprocessed_docs, doc_embs, targets, vocabularys, device


# In[4]:


unpreprocessed_docs, preprocessed_docs, doc_embs, targets, vocabularys, device = load_training_data(config, config['dataset'])


# In[5]:


if config['dataset2'] is not None:
    unpreprocessed_docs2, preprocessed_docs2, doc_embs2, targets2, vocabularys2, device = load_training_data(config, config['dataset2'])
    targets, targets2, vocabularys = merge_targets(targets, targets2, vocabularys, vocabularys2)
    


# In[6]:


word_embs = get_word_embs(vocabularys)
print('word_embs', word_embs.shape)
word_embs_tensor = torch.FloatTensor(word_embs)


# ## MLP Decoder

# In[7]:


class DNNDecoderDataset(Dataset):
    def __init__(self, doc_embs, targets):
        
        assert len(doc_embs) == len(targets)

        self.doc_embs = torch.FloatTensor(doc_embs)
        self.targets = torch.FloatTensor(targets)        
        self.targets_rank = torch.argsort(self.targets, dim=1, descending=True)
        self.topk = torch.sum(self.targets > 0, dim=1)
        
    def __getitem__(self, idx):
        return self.doc_embs[idx], self.targets[idx], self.targets_rank[idx], self.topk[idx]

    def __len__(self):
        return len(self.doc_embs)


# In[8]:


def prepare_dataloader(doc_embs, targets, no_split=False, batch_size=100, train_size_ratio=0.8, target_normalize=False, seed=123):
    train_size = int(len(doc_embs) * train_size_ratio)
    test_size = len(doc_embs) - train_size

    print('Preparing dataloader')
    print('train size', train_size)
    print('test size', test_size)

    if target_normalize:
        # normalize target summation of each document to 1 
        norm = targets.sum(axis=1).reshape(-1, 1)
        targets = (targets / norm)
        # normalize target L2 norm of each document to 1
        # norm = np.linalg.norm(targets, axis=1).reshape(-1, 1)
        # targets = (targets / norm)

    dataset = DNNDecoderDataset(doc_embs, targets)
    train_dataset, test_dataset = random_split(dataset, lengths=[train_size, test_size],generator=torch.Generator().manual_seed(42))

    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_loader, valid_loader, test_loader


# In[9]:


# prepare dataloader
if config['dataset2'] is None:
    train_loader, valid_loader, test_loader = prepare_dataloader(doc_embs, targets, batch_size=64, train_size_ratio=config['train_size'], \
                                                                target_normalize=config['target_normalization'], seed=seed)
else:
    train_loader, valid_loader, _ = prepare_dataloader(doc_embs, targets, batch_size=64, train_size_ratio=0.99, \
                                                                target_normalize=config['target_normalization'], seed=seed)
    test_loader, _, _ = prepare_dataloader(doc_embs2, targets2, batch_size=64, train_size_ratio=0.99, \
                                           target_normalize=config['target_normalization'], seed=seed)


class DNNDecoder(nn.Module):

    ### casimir
    # (1) Add parameter vocab_size
    def __init__(self, doc_emb_dim, num_words=0, h_dim=300, dropout=0.2):
        super(DNNDecoder, self).__init__()
        vocab_size = num_words
        bert_size = doc_emb_dim
        
        self.vocab_size = vocab_size
        if dropout > 0:
            self.network = nn.Sequential(
            nn.Linear(bert_size, bert_size*4),
            nn.BatchNorm1d(bert_size*4),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
            nn.Linear(bert_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
#             nn.Sigmoid(),
            )
        else:
            self.network = nn.Sequential(
            nn.Linear(bert_size, bert_size*4),
            nn.BatchNorm1d(bert_size*4),
            nn.Sigmoid(),
            nn.Linear(bert_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
#             nn.Sigmoid(),
            )
        
    def forward(self, x_bert):
        recon_dist = self.network(x_bert)

        return recon_dist
# In[11]:


def evaluate_DNNDecoder(model, data_loader, config, pred_semantic=False):
    results = defaultdict(list)
    model.eval()
    
    # predict all data
    for data in data_loader:
        doc_embs, target, _, _ = data
        
        doc_embs = doc_embs.to(device)
        target = target.to(device)
                
        pred = model(doc_embs)
        if config['eval_f1']:
            # Precision / Recall / F1
            p, r, f = precision_recall_f1_all(pred, target)
            results['precision'].append(p)
            results['recall'].append(r)
            results['f1_score'].append(f)
        else:
            # Precision
            precision_scores = retrieval_precision_all(pred, target, k=config["valid_topk"])
            for k, v in precision_scores.items():
                results['precision@{}'.format(k)].append(v)

            # Precision
            precision_scores = retrieval_precision_all_v2(pred, target, k=config["valid_topk"])
            for k, v in precision_scores.items():
                results['precisionv2@{}'.format(k)].append(v)

            # NDCG
            ndcg_scores = retrieval_normalized_dcg_all(pred, target, k=config["valid_topk"])
            for k, v in ndcg_scores.items():
                results['ndcg@{}'.format(k)].append(v)
            
            # Semantic Precision
            if pred_semantic:
                semantic_precision_scores, word_result = semantic_precision_all(pred, target, word_embs_tensor, vocabularys,                                                                                k=config["valid_topk"], th=0.5, display_word_result=False)
                for k, v in semantic_precision_scores.items():
                    results['semantic_precision@{}'.format(k)].append(v)
                    
                semantic_precision_scores, word_result = semantic_precision_all_v2(pred, target, word_embs_tensor, vocabularys,                                                                                k=config["valid_topk"], th=0.5, display_word_result=False)
                for k, v in semantic_precision_scores.items():
                    results['semantic_precision_v2@{}'.format(k)].append(v)

    for k in results:
        results[k] = np.mean(results[k])

    return results


# In[12]:


def calculate_loss(train_train_config, criterion, pred, target, target_rank, target_topk):
    if train_config["criterion"] == "MultiLabelMarginLoss":
        assert target_rank.shape[0] == len(target_topk)
        for i in range(len(target_topk)):
            target_rank[i, target_topk[i]] = -1
        loss = criterion(pred, target_rank)
    elif train_config["criterion"].startswith("MultiLabelMarginLossCustomV"):
        loss = criterion(pred, target_rank, target_topk)
    elif train_config["criterion"].startswith("MultiLabelMarginLossCustom"):
        loss = criterion(pred, target_rank, train_config["loss_topk"])
    else:
        loss = criterion(pred, target)
        
    return loss
    
def train_decoder(doc_embs, targets, train_config):
    model = DNNDecoder(doc_emb_dim=doc_embs.shape[1], num_words=targets.shape[1], h_dim=train_config["h_dim"], dropout=config['dropout']).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])
    # prepare loss
    if train_config["criterion"] == "MultiLabelMarginLoss":
        criterion = nn.MultiLabelMarginLoss(reduction='mean')
    elif train_config["criterion"] == "BCE":
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif train_config["criterion"].startswith("MultiLabelMarginLossCustomV"):
        def criterion(a, b, c): return MultiLabelMarginLossCustomV(
            a, b, c, float(train_config["criterion"].split(':')[-1]))
    elif train_config["criterion"].startswith("MultiLabelMarginLossCustom"):
        def criterion(a, b, c): return MultiLabelMarginLossCustom(
            a, b, c, float(train_config["criterion"].split(':')[-1]))
    else:
        criterion = eval(train_config["criterion"])

    results = []
    n_epoch = train_config["n_epoch"]
    valid_epoch = train_config["valid_epoch"]
    valid_verbose = train_config["valid_verbose"]

    for epoch in tqdm(range(n_epoch)):
        train_loss_his = []
        valid_loss_his = []

        model.train()

        for data in train_loader:
            doc_embs, target, target_rank, target_topk = data
            doc_embs = doc_embs.to(device)
            target = target.to(device)
            target_rank = target_rank.to(device)
            target_topk = target_topk.to(device)
            # loss
            pred = model(doc_embs)
            loss = calculate_loss(train_config, criterion, pred, target, target_rank, target_topk)
            train_loss_his.append(loss.item())

            # Model backwarding
            model.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        for data in valid_loader:
            doc_embs, target, target_rank, target_topk = data
            doc_embs = doc_embs.to(device)
            target = target.to(device)
            target_rank = target_rank.to(device)
            target_topk = target_topk.to(device)

            # loss
            pred = model(doc_embs)
            loss = calculate_loss(train_config, criterion, pred, target, target_rank, target_topk)
            valid_loss_his.append(loss.item())

        print("Epoch", epoch, np.mean(train_loss_his), np.mean(valid_loss_his))

        # show decoder result
        if (valid_epoch > 0 and epoch % valid_epoch == 0) or epoch == n_epoch-1:
            res = {}
            res['epoch'] = epoch

            train_res_ndcg = evaluate_DNNDecoder(model, train_loader, train_config, False)
            valid_res_ndcg = evaluate_DNNDecoder(model, valid_loader, train_config, False)
            test_res_ndcg = evaluate_DNNDecoder(model, test_loader, train_config, False)
            
            res['train'] = train_res_ndcg
            res['valid'] = valid_res_ndcg
            res['test'] = test_res_ndcg 
            results.append(res)

            if valid_verbose:
                print()
                print('train', train_res_ndcg)
                print('valid', valid_res_ndcg)
                print('test', test_res_ndcg)
    return results

def train_experiment(n_time):
    # train n_time in different seed
    results = []
    for _ in range(n_time):
        result = train_decoder(doc_embs, targets, train_config)
        results.append(result)

    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(results, f)

    return results


# In[13]:


train_config = {
    "n_time": config['n_time'],
    "lr": config['lr'],
    "weight_decay": 0.0,
    "loss_topk": 15,
    "dropout": config['dropout'],

    "n_epoch": config['n_epoch'],
    "valid_epoch": config['valid_epoch'],
    "valid_verbose": True,
    "valid_topk": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    
    "h_dim": config['h_dim'],
    "label_type": config['label_type'],
    "eval_f1": config['eval_f1'],
    "criterion": config['criterion']
}


# In[ ]:


train_experiment(train_config['n_time'])


# In[ ]:


# save config, training config
with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(config, f)
with open(os.path.join(save_dir, 'train_config.json'), 'w') as f:
    json.dump(train_config, f)

