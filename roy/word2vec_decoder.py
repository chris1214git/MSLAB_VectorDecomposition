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
from collections import defaultdict, Counter
from itertools import cycle
from data_utils import IDEDataset, WordEmbeddingDataset
from models import W2vDecoder

sys.path.append("../")

from utils.data_processing import get_process_data
from utils.data_loader import load_document
from utils.loss import ListNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, get_freer_gpu, show_settings

# config
parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="IMDB")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default=get_freer_gpu())
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
args = parser.parse_args()
config = vars(args)

show_settings(config)

same_seeds(config["seed"])

device = config["gpu"]

# load data
dataset = "IMDB"
docvec = np.load("../data/docvec_IMDB_SBERT_768d.npy")
dim = 768
raw_documents = load_document(dataset)["documents"]

# preprocess
# get TF-IDF score
vectorizer = TfidfVectorizer(min_df=10,stop_words="english")
importance_score = np.array(vectorizer.fit_transform(raw_documents).todense())

vocab_size = len(vectorizer.vocabulary_)
itos = vectorizer.get_feature_names()
stoi = vectorizer.vocabulary_
vocabulary = vectorizer.get_feature_names()
print(f"Vocab size:{vocab_size}")

# re-ordering words in document
# important words would be place at the beginning
# using word2vec as pretrained word embedding
PATTERN = r"(?u)\b\w\w+\b"
tokenized_documents = [re.findall(PATTERN, i.lower()) for i in raw_documents]
ranked_wordID = np.argsort(np.array(importance_score))

ranked_documents = []
preserve_topk = 50
for doc_id,doc in enumerate(tqdm(tokenized_documents)):
    doc_length = len(doc)
    doc_length = min(doc_length,preserve_topk)
    words = [itos[int(i)] for i in ranked_wordID[doc_id][-doc_length:]]
    ranked_documents.append(words)

numericalize_document = []
for doc in tqdm(ranked_documents):
    tmp = [stoi[w] for w in doc if w in stoi]
    numericalize_document.append(tmp)

word_freqs = np.ones(vocab_size) # uniform negative sampling distribution

# datasets
dataset = IDEDataset(docvec, importance_score)
train_length = int(len(dataset)*0.6)
valid_length = int(len(dataset)*0.2)
test_length = len(dataset) - train_length - valid_length

full_loader = DataLoader(dataset, batch_size=128)
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, lengths=[train_length, valid_length,test_length],
    generator=torch.Generator().manual_seed(42)
    )

# loader 
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], 
    shuffle=True, pin_memory=True,
    num_workers = 4,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers = 4,)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False,num_workers = 4,)

# word2vec dataset
word2vec_dataset = WordEmbeddingDataset(numericalize_document,word_freqs)
word2vec_loader = DataLoader(word2vec_dataset, 512, shuffle=True, pin_memory=True,drop_last=True,num_workers = 4,)
word2vec_iterloader = cycle(word2vec_loader)

# eval
def evaluate_Decoder(model, data_loader):
    results = defaultdict(list)
    model.eval()
        
    # predict all data
    for data in data_loader:
        doc_embs, target, _, _ = data
        
        doc_embs = doc_embs.to(device)
        target = target.to(device)
                
        pred = model(doc_embs)
    
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


# init models
decoder = W2vDecoder(input_dim=dim,hidden_dim=1024,output_dim=vocab_size).to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr = 1e-4)
word2vec_optimizer = torch.optim.Adam(decoder.word2vec.parameters(), lr = 1e-3,weight_decay=1e-5)

# initialize parameters
for p in decoder.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# training
# early stop settings 
stop_rounds = 3
no_improvement = 0
best_score = None 


for epoch in range(100):
    # Training
    decoder.train()
    train_loss = []
    avg_rank_loss = []
    avg_w2v_loss = []
    for batch in tqdm(train_loader, desc="Training"):
        batch = [i.to(device) for i in batch]
        doc_embs, target, _, _ = batch
        target = torch.nn.functional.normalize(target.to(device), dim=1)
        decoded = torch.nn.functional.normalize(decoder(doc_embs), dim=1)
        rank_loss = ListNet(decoded, target)
        
        # word2vec part
        input_labels, pos_labels, neg_labels = next(word2vec_iterloader)
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)
        w2v_loss = 0.1 * decoder.word2vec_loss(input_labels, pos_labels, neg_labels).mean()
        
        loss = rank_loss + w2v_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        avg_w2v_loss.append(w2v_loss.item())
        avg_rank_loss.append(rank_loss.item())

    print(f"[Epoch {epoch+1:02d}]")
    print(f"Loss:{np.mean(train_loss):.4f} Rank Loss:{np.mean(avg_rank_loss):.4f} Word2vec Loss: {np.mean(avg_w2v_loss):.4f}")
    res = evaluate_Decoder(decoder, valid_loader)
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


print("Testing...")
res = evaluate_Decoder(decoder, test_loader)
for key,val in res.items():
    print(f"{key}:{val:.4f}")