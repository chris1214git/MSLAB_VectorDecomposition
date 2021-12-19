import os
import re
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../..")

# from model import Decoder_only, Decoder_wordembed
from utils.loss import ListNet, MythNet
from utils.data_processing import get_process_data
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, get_freer_gpu, show_settings, split_data, doc_filter, generate_graph

def evaluate_Decoder(model, data_loader, config):
    results = defaultdict(list)
    model.eval()
        
    # predict all data
    for data in data_loader:
        doc_embs, target = data
        
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

class IDEDataset(Dataset):
    def __init__(self, doc_emb, doc_tfidf):
        
        assert len(doc_emb) == len(doc_tfidf)
        self.doc_emb = torch.FloatTensor(doc_emb)
        self.doc_tfidf = torch.FloatTensor(doc_tfidf)        
        
    def __getitem__(self, idx):
        return self.doc_emb[idx], self.doc_tfidf[idx]

    def __len__(self):
        return len(self.doc_emb)

# Pytorch Geometric Package
class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = GCNConv(input_dim, output_dim)
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class GraphAE_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, output_dim):
        super().__init__()
        self.word_embedding = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.autoencoder = VGAE(VariationalGCNEncoder(hidden_dim, encoded_dim))
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        docvec = self.decoder(x)
        decoded = torch.sigmoid(torch.matmul(docvec, self.word_embedding))
        return decoded
    
    def graph_loss(self, edge_index):
        encoded = self.autoencoder.encode(torch.transpose(self.word_embedding, 0, 1), edge_index)
        loss = self.autoencoder.recon_loss(encoded, edge_index)
        return loss    

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="IMDB")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default=get_freer_gpu())
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--encoded_dim', type=int, default=1024)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
    args = parser.parse_args()
    
    config = vars(args)
    show_settings(config)
    same_seeds(config["seed"])
    device = config["gpu"]

    data_dict = get_process_data(config['dataset'])
    doc_raw = data_dict['dataset']['documents']
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=10, stop_words="english")
    doc_tfidf = vectorizer.fit_transform(doc_raw).todense()
    doc_embedding = np.load('./docvec_IMDB_SBERT_768d.npy')
    
    # Build Vocabulary set & delete word based on vocabulary
    word2index = vectorizer.vocabulary_
    vocab_size = len(word2index)
    vocab_set = set(word2index)
    index2word = {}
    for i in word2index:
        index2word[word2index[i]] = i
    doc_list = [doc_filter(doc, vocab_set) for doc in tqdm(doc_raw, desc="Delete word from raw document:")]
     
    # Build Graph
    edge_index = torch.tensor(generate_graph(doc_list, word2index, index2word), dtype=torch.long)
    
    # Prepare dataset
    dataset = IDEDataset(doc_embedding, doc_tfidf)
    train_length = int(len(dataset)*config['ratio'])
    valid_length = int(len(dataset)*(0.8-config['ratio']))
    test_length = len(dataset) - train_length - valid_length
    full_loader = DataLoader(dataset, batch_size=config['batch_size'])
    train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths=[train_length, valid_length,test_length], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Declare model
    decoder = GraphAE_Decoder(input_dim=768, hidden_dim=config['hidden_dim'], encoded_dim=config['encoded_dim'], output_dim=vocab_size)
    optimizer = torch.optim.Adam(decoder.parameters(), lr = config['lr'])
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    decoder = decoder.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(config['epochs']):
        # Training
        decoder.train()
        total_train_loss = 0
        update_ae = True
        for batch, (doc_emb, doc_tfidf) in enumerate(tqdm(train_loader, desc="Training")):
            doc_emb, doc_tfidf = doc_emb.to(device), torch.nn.functional.normalize(doc_tfidf.to(device), dim=1)
            decoded = torch.nn.functional.normalize(decoder(doc_emb), dim=1)
            loss = ListNet(decoded, doc_tfidf)
            if random.random() <= 0.9:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                update_ae = False
                ae_loss = decoder.graph_loss(edge_index.t().contiguous())
                loss += ae_loss * 0.1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_train_loss += loss.item()
        print(f"[Epoch {epoch+1:02d}]")
        res = evaluate_Decoder(decoder, test_loader, config)
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
        print('Decode Loss: {}\nReconstruct Loss: {}'.format(total_train_loss, ae_loss.item()))
        print('----------------------------------------')