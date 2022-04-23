import os
import re
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_cluster import random_walk
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv
from torch_geometric.data import Data
from tqdm.auto import tqdm
from itertools import cycle
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../..")

# from model import Decoder_only, Decoder_wordembed
from utils.loss import ListNet, MythNet
from utils.data_processing import get_process_data
from utils.data_loader import load_document
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

class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class GraphAE_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim, output_dim):
        super().__init__()
        self.word_embedding = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.autoencoder = SAGE(in_channels=hidden_dim, hidden_channels=256, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Linear(input_dim*4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        docvec = self.decoder(x)
        decoded = torch.sigmoid(torch.matmul(docvec, self.word_embedding))
        return decoded
    
    def graph_out(self, n_id, adjs):
        output = self.autoencoder(torch.transpose(self.word_embedding, 0, 1)[n_id], adjs)
        return output 
    def get_word_emb(self):
        return self.word_embedding

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="IMDB")
    parser.add_argument('--target', type=str, default='tfidf')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='free')
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
    if config['gpu'] == 'multi':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Use {} GPUs for training'.format(torch.cuda.device_count()))
    else:
        device = get_freer_gpu()

    # data_dict = get_process_data(config['dataset'])
    # doc_raw = data_dict['dataset']['documents']
    data_dict = load_document(config['dataset'])
    doc_raw = data_dict['documents']
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=10, stop_words="english")
    doc_tfidf = vectorizer.fit_transform(doc_raw).todense()

    if config['target'] == 'keyBERT':
        # KeyBERT
        doc_target_sparse = csr_matrix(np.load('../../../data1/IDE/precompute_keyword/keyword_IMDB_KeyBERT.npy'))
        doc_target = np.array(doc_target_sparse.todense())
        del doc_tfidf
    else:
        # TF-IDF
        doc_target = np.array(doc_tfidf)
    
    doc_embedding = np.load('./docvec_IMDB_SBERT_768d.npy')
    doc_emb_dim = doc_embedding.shape[1]
    
    # Build Vocabulary set & delete word based on vocabulary
    word2index = vectorizer.vocabulary_
    vocab_size = len(word2index)
    vocab_set = set(word2index)
    index2word = {}
    for i in word2index:
        index2word[word2index[i]] = i
    doc_list = [doc_filter(doc, vocab_set) for doc in tqdm(doc_raw, desc="Delete word from raw document:")]

    # Sort Document
    doc_ranks = np.zeros((doc_tfidf.shape[0], len(word2index)), dtype='float32')
    for i in range(doc_tfidf.shape[0]):
        doc_ranks[i] = np.argsort(doc_tfidf[i])[::-1]

    doc_sort_list = []
    for i in range(doc_tfidf.shape[0]):
        doc_sort = []
        for j in range(len(doc_list[i])):
            doc_sort.append(index2word[int(doc_ranks[i][j])])
        doc_sort_list.append(doc_sort)
    # Build Graph
    edge_index = torch.tensor(generate_graph(doc_list, word2index, index2word), dtype=torch.long).t().contiguous()
    
    # Prepare dataset
    dataset = IDEDataset(doc_embedding, doc_target)
    train_length = int(len(dataset)*config['ratio'])
    valid_length = int(len(dataset)*(0.8-config['ratio']))
    test_length = len(dataset) - train_length - valid_length
    full_loader = DataLoader(dataset, batch_size=config['batch_size'])
    train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths=[train_length, valid_length,test_length], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    graph_loader = NeighborSampler(edge_index, sizes=[10, 10], batch_size=config['batch_size'], shuffle=True, num_nodes=len(word2index))
    graph_iterloader = cycle(graph_loader)
    # Declare model
    decoder = GraphAE_Decoder(input_dim=doc_emb_dim, hidden_dim=config['hidden_dim'], encoded_dim=config['encoded_dim'], output_dim=vocab_size)
    if config['gpu'] == 'multi':
        decoder = nn.DataParallel(decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr = config['lr'])
    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    decoder = decoder.to(device)
    edge_index = edge_index.to(device)

    # Training
    for epoch in range(config['epochs']):
        decoder.train()
        total_train_loss = 0
        for batch, (doc_emb, doc_tfidf) in enumerate(tqdm(train_loader, desc="Training")):
            # ListNet Decoder
            doc_emb, doc_tfidf = doc_emb.to(device), torch.nn.functional.normalize(doc_tfidf.to(device), dim=1)
            decoded = torch.nn.functional.normalize(decoder(doc_emb), dim=1)
            de_loss = ListNet(decoded, doc_tfidf)
        
            # SAGE
            batch_size, n_id, adjs = next(graph_iterloader)
            adjs = [adj.to(device) for adj in adjs]
            ae_output = decoder.graph_out(n_id, adjs)
            out, pos_out, neg_out = ae_output.split(ae_output.size(0) // 3, dim=0)  
            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            ae_loss = -pos_loss - neg_loss
        
            # loss
            loss = de_loss + ae_loss
            loss.backward()
            optimizer.step()
            #graph_optimizer.step()
            optimizer.zero_grad()
            #graph_optimizer.zero_grad()
        
            total_train_loss += loss.item()
        print(f"[Epoch {epoch+1:02d}]")
        res = evaluate_Decoder(decoder, test_loader, config)
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
        print('Decode Loss: {}\nReconstruct Loss: {}'.format(total_train_loss, ae_loss.item()))
        print('----------------------------------------')