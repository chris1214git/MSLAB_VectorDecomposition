#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import argparse
import sys
from gensim.models import Word2Vec
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict,OrderedDict
from contextualized_topic_models.models.ctm import ZeroShotTM

sys.path.append("../")

from utils.data_processing import get_process_data
from utils.data_loader import load_document
from utils.loss import ListNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, get_freer_gpu


# In[2]:


device = f"cuda:{get_freer_gpu()}"
config = {
    "topk":[10,30,50]
}


# In[3]:


seed = 123
seed_everything(123)


# In[4]:


dataset = "IMDB"
docvec = np.load("../data/docvec_IMDB_SBERT_768d.npy")
dim = 768
raw_documents = load_document(dataset)["documents"]


# In[5]:


# get TF-IDF score
vectorizer = TfidfVectorizer(min_df=10,stop_words="english")
importance_score = vectorizer.fit_transform(raw_documents).todense()


# In[6]:


vocab_size = len(vectorizer.vocabulary_)
print(f"Vocab size:{vocab_size}")


# In[7]:


class IDEDataset(Dataset):
    def __init__(self, 
                 doc_vectors,
                 weight_ans):
        
        assert len(doc_vectors) == len(weight_ans)

        self.doc_vectors = torch.FloatTensor(doc_vectors)
        self.weight_ans = torch.FloatTensor(weight_ans)        
        self.weight_ans_s = torch.argsort(self.weight_ans, dim=1, descending=True)
        self.topk = torch.sum(self.weight_ans > 0, dim=1)
        
    def __getitem__(self, idx):
        return self.doc_vectors[idx], self.weight_ans[idx], self.weight_ans_s[idx], self.topk[idx]

    def __len__(self):
        return len(self.doc_vectors)


# In[8]:


dataset = IDEDataset(docvec, importance_score)
train_length = int(len(dataset)*0.6)
valid_length = int(len(dataset)*0.2)
test_length = len(dataset) - train_length - valid_length

full_loader = DataLoader(dataset, batch_size=128)
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, lengths=[train_length, valid_length,test_length],
    generator=torch.Generator().manual_seed(42)
    )


# In[9]:


train_loader = DataLoader(
    train_dataset, batch_size=128, 
    shuffle=True, pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=128, shuffle=False, pin_memory=True,drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False)


# In[24]:


def evaluate_Decoder(model, data_loader):
    results = defaultdict(list)
    model.eval()
        
    # predict all data
    for data in data_loader:
        doc_embs, target, _, _ = data
        
        doc_embs = doc_embs.to(device)
        target = target.to(device)
                
        pred = model(target,doc_embs)
        pred = pred[-2]
    
        # Precision
        precision_scores = retrieval_precision_all(pred, target, k=config["topk"])
        for k, v in precision_scores.items():
            results['precision@{}'.format(k)].append(v)
        
        # NDCG
        ndcg_scores = retrieval_normalized_dcg_all(pred, target, k=config["topk"])
        for k, v in ndcg_scores.items():
            results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results


# In[54]:


# decoder = Decoder(input_dim=dim,hidden_dim=1024,output_dim=vocab_size)
model = ZeroShotTM(bow_size=vocab_size, contextual_size=768,n_components=1024,hidden_sizes=(1024,1024),activation="relu")
decodernet = model.model
# optimizer = torch.optim.Adam(decodernet.parameters(), lr = 1e-3)
optimizer = model.optimizer
# initialize parameters
for p in decodernet.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# decoder.load_pretrianed(pretrain_wordembedding)
decodernet = decodernet.to(device)


# In[55]:


# early stop settings 
stop_rounds = 300
no_improvement = 0
best_score = None 


for epoch in range(100):
    # Training
    decodernet.train()
    train_loss = []
    for batch in tqdm(train_loader, desc="Training"):
        batch = [i.to(device) for i in batch]
        doc_embs, target, _, _ = batch
        prior_mean, prior_variance, posterior_mean, posterior_variance,            posterior_log_variance, word_dists,_ = decodernet(target,doc_embs)
        kl_loss, rl_loss = model._loss(target,word_dists, prior_mean, prior_variance,
                        posterior_mean, posterior_variance, posterior_log_variance)
        loss = 1e-3*kl_loss + rl_loss
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())

    print(f"[Epoch {epoch+1:02d}]")
    res = evaluate_Decoder(decodernet, valid_loader)
    for key,val in res.items():
        print(f"{key}:{val:.4f}")
        
     # early stopping 
    current_score = res["precision@10"]
    if best_score == None:
        best_score = current_score
        continue
    if current_score < best_score:
        no_improvement += 1
    if no_improvement >= stop_rounds:
        print("Early stopping...")
        break 
    if current_score > best_score:
        no_improvement = 0
        best_score = current_score


# In[ ]:


print("Testing...")
res = evaluate_Decoder(decoder, test_loader)
for key,val in res.items():
    print(f"{key}:{val:.4f}")


# In[ ]:




