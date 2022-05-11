#!/usr/bin/env python
# coding: utf-8

# # Demo baseline
# 
# ### document embedding decoder
# 1. demo utils
# 2. demo loss
# 3. demo evaluation

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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, retrieval_precision_all_v2, semantic_precision_all, semantic_precision_all_v2, precision_recall_f1_all
from utils.loss import *
from utils.data_loader import load_document
from utils.toolbox import preprocess_document, get_preprocess_document, get_preprocess_document_embs,                          get_preprocess_document_labels, get_preprocess_document_labels_v2, get_word_embs,                          get_free_gpu, merge_targets


# ## Data preprocess
# 1. filter special characters, punctuation (remain english & number character)
# 2. filter stopwords
# 3. filter by term frequency
# 4. pos tagging

# ## Parameters
# 
# ### preprocess parameters:
# 1. min word frequency
# 2. max word frequency(max_df)
# 3. min word per doc(min_words)
# 4. pos tagging select
# 
# ### training parameters:
# 1. decoder label
# 2. model parameters

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
if dataset2:
    experiment_dir = f'cross_{dataset}_{dataset2}_{model_name}_{label_type}_{criterion}'
else:
    experiment_dir = f'{dataset}_{model_name}_{label_type}_{criterion}'
    
save_dir = 'default'

config = {}
config['experiment_dir'] = experiment_dir
config['preprocess_config_dir'] = preprocess_config_dir
config['save_dir'] = save_dir
config['dataset'] = dataset
config['dataset2'] = dataset2
config['model_name'] = model_name
config['label_type'] = label_type
config['eval_f1'] = eval_f1
config['n_gram'] = n_gram
config['criterion'] = criterion
config['n_time'] = n_time
config['seed'] = seed

save_dir = os.path.join('experiment', config['experiment_dir'], config['save_dir'])
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


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


# In[8]:


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


# In[10]:


def prepare_dataloader(doc_embs, targets, batch_size=100, train_valid_test_ratio=[0.7, 0.1, 0.2],                       target_normalize=False, seed=123):
    train_size = int(len(doc_embs) * train_valid_test_ratio[0])
    valid_size = int(len(doc_embs) * (train_valid_test_ratio[0] + train_valid_test_ratio[1])) - train_size
    test_size = len(doc_embs) - train_size - valid_size
    
    print('Preparing dataloader')
    print('train size', train_size)
    print('valid size', valid_size)
    print('test size', test_size)

    if target_normalize:
        # normalize target summation of each document to 1 
        norm = targets.sum(axis=1).reshape(-1, 1)
        targets = (targets / norm)
        # normalize target L2 norm of each document to 1
        # norm = np.linalg.norm(targets, axis=1).reshape(-1, 1)
        # targets = (targets / norm)

    # shuffle
    randomize = np.arange(len(doc_embs))
    np.random.seed(seed)
    np.random.shuffle(randomize)
    doc_embs = doc_embs[randomize]
    targets = targets[randomize]
    
    # dataloader
    train_dataset = DNNDecoderDataset(doc_embs[:train_size], targets[:train_size])
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = DNNDecoderDataset(doc_embs[train_size:train_size+valid_size], targets[train_size:train_size+valid_size])
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_dataset = DNNDecoderDataset(doc_embs[train_size+valid_size:], targets[train_size+valid_size:])
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return train_loader, valid_loader, test_loader


# In[11]:


# prepare dataloader
train_loader, valid_loader, test_loader = prepare_dataloader(doc_embs, targets, batch_size=64,                                                             train_valid_test_ratio=[0.7, 0.1, 0.2],target_normalize=True,                                                             seed=seed)
if config['dataset2'] is not None:
    _, _, test_loader = prepare_dataloader(doc_embs2, targets2, batch_size=64,                                                             train_valid_test_ratio=[0.7, 0.1, 0.2],target_normalize=True,                                                             seed=seed)


# In[12]:


class DNNDecoder(nn.Module):
    def __init__(self, doc_emb_dim, num_words, h_dim=300):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(doc_emb_dim, h_dim),
            # nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            # nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(h_dim, num_words),
            # nn.Dropout(p=0.5),
            # nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(x)


# In[13]:


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
                semantic_precision_scores, word_result = semantic_precision_all(pred, target, word_embs_tensor, vocabularys,\
                                                                                k=config["valid_topk"], th=0.5, display_word_result=False)
                for k, v in semantic_precision_scores.items():
                    results['semantic_precision@{}'.format(k)].append(v)
                    
                semantic_precision_scores, word_result = semantic_precision_all_v2(pred, target, word_embs_tensor, vocabularys,\
                                                                                k=config["valid_topk"], th=0.5, display_word_result=False)
                for k, v in semantic_precision_scores.items():
                    results['semantic_precision_v2@{}'.format(k)].append(v)

    for k in results:
        results[k] = np.mean(results[k])

    return results

# In[15]:


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
    model = DNNDecoder(doc_emb_dim=doc_embs.shape[1], num_words=targets.shape[1],                       h_dim=train_config["h_dim"]).to(device)
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
            y_pos_id = target_rank[:, :4]
            y_neg_id = target_rank[:, 4:]
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

            train_res_ndcg = evaluate_DNNDecoder(model, train_loader, train_config, epoch == n_epoch-1)
            valid_res_ndcg = evaluate_DNNDecoder(model, valid_loader, train_config, epoch == n_epoch-1)
            test_res_ndcg = evaluate_DNNDecoder(model, test_loader, train_config, epoch == n_epoch-1)
            
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


# In[16]:


train_config = {
    "n_time": config['n_time'],
    "lr": 0.001,
    "weight_decay": 0.0,
    "loss_topk": 15,
    
    "n_epoch": 1000,
    "valid_epoch": 10,
    "valid_verbose": True,
    "valid_topk": [5, 10, 15],
    
    "h_dim": 300,
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


# ## Result
# Run 5 times, different model seed, same train/valid/test split, mean/std
# 1. precision, recall, f1
# 2. precision, ndcg, semantic precision
# 
# Exp:
# 1. different doc encoder
# 2. different dataset(mpnet)
# 3. cross domain(mpnet)
# 4. different target(mpnet, agnews)(bow, tf-idf, keybert, yake)

# * bow:
#     3 dataset * bce * 4 models
# * tf-idf:
#     3 dataset * listnet * 4 models
# * keybert, yake:
#     agnews * listnet * 4 models
# * cross domain
