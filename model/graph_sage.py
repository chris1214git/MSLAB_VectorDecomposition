import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_cluster import random_walk
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import GCNConv, GAE, VGAE, SAGEConv, GATConv
from torch_geometric.data import Data
from tqdm.auto import tqdm
from itertools import cycle

from utils.loss import ListNet, MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all
from utils.toolbox import get_free_gpu, record_settings

class GraphSAGE_Dataset(Dataset):
    def __init__(self, corpus, emb, target):
        
        assert len(corpus) == len(emb)
        self.corpus = corpus
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)        
        
    def __getitem__(self, idx):
        return self.corpus[idx], self.emb[idx], self.target[idx]

    def __len__(self):
        return len(self.corpus)

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
            self.convs.append(GATConv(in_channels, hidden_channels))

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
    
class DecoderNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.word_embedding = nn.Parameter(torch.randn(output_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.sage = SAGE(in_channels=output_dim, hidden_channels=256, num_layers=2)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        docvec = self.decoder(x)
        decoded = torch.sigmoid(self.batch_norm(torch.matmul(docvec, self.word_embedding)))
        return decoded
    
    def graph_update(self, n_id, adjs):
        output = self.sage(torch.transpose(self.word_embedding, 0, 1)[n_id], adjs)
        return output 
    def get_word_emb(self):
        return self.word_embedding

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.output_dim = output_dim
       
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.network(x)
        return decoded

class GraphSAGE:
    def __init__(self, config, edge_index=None, vocabulary=None, id2token=None, contextual_size=768, vocab_size=8000, word_embeddings=None):
        if torch.cuda.is_available():
            self.device = get_free_gpu()
        else:
            self.device = torch.device("cpu")
        self.config = config
        self.edge_index = edge_index
        self.vocabulary = vocabulary
        self.id2token = id2token
        self.contextual_size = contextual_size
        self.vocab_size = vocab_size
        self.num_epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.word_embeddings = word_embeddings

        if config['model'] == 'GraphSAGE':
            self.decoder = DecoderNetwork(input_dim=contextual_size, output_dim=vocab_size)
        else:
            self.decoder = MLPNetwork(input_dim=contextual_size, output_dim=vocab_size)

        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas=(self.momentum, 0.99))
        elif config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.lr, momentum=self.momentum)

    def fit(self, training_set, validation_set):
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        if self.config['model'] == 'GraphSAGE':
            graph_loader = NeighborSampler(self.edge_index, sizes=[25, 10], batch_size=self.batch_size, shuffle=True, num_nodes=self.vocab_size)
            graph_iterloader = cycle(graph_loader)

        self.decoder = self.decoder.to(self.device)
        if self.config['model'] == 'GraphSAGE':
            self.edge_index = self.edge_index.to(self.device)

        for epoch in range(self.num_epochs):
            self.decoder.train()
            for batch, (corpus, emb, target) in enumerate(tqdm(training_loader, desc="Training")):
                # MLP Decoder
                emb, target = emb.to(self.device), target.to(self.device)
                de_loss = MythNet(self.decoder(emb), target)

                if self.config['model'] == 'GraphSAGE':
                # SAGE
                    batch_size, n_id, adjs = next(graph_iterloader)
                    adjs = [adj.to(self.device) for adj in adjs]
                    sage_output = self.decoder.graph_update(n_id, adjs)
                    out, pos_out, neg_out = sage_output.split(sage_output.size(0) // 3, dim=0)  
                    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
                    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
                    sage_loss = -pos_loss - neg_loss
                    loss = de_loss + sage_loss
                else:
                    loss = de_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if  (epoch + 1) % 10 == 0:
                validation_result = self.validation(validation_loader)
                record = open('./'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['target']+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print('EPOCH', epoch + 1)
                record.write('EPOCH '+ str(epoch + 1) + '\n')
                for key,val in validation_result.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")

        if self.config['visualize']:
            self.visualize(validation_set, validation_loader)

    def validation(self, loader):
        self.decoder.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (corpus, emb, target) in enumerate(loader):
                emb, target = emb.to(self.device), target.to(self.device)
                recon_dists = self.decoder(emb)
                # Semantic Prcision for reconstruct
                precision_scores, word_result = semantic_precision_all(recon_dists, target, self.word_embeddings, self.vocabulary, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    results['[Recon] Semantic Precision@{}'.format(k)].append(v)
                    
                # Precision for reconstruct
                precision_scores = retrieval_precision_all(recon_dists, target, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision@{}'.format(k)].append(v)

                # NDCG for reconstruct
                ndcg_scores = retrieval_normalized_dcg_all(recon_dists, target, k=self.config['topk'])
                for k, v in ndcg_scores.items():
                    results['[Recon] ndcg@{}'.format(k)].append(v)
            for k in results:
                results[k] = np.mean(results[k])
        return results

    def get_reconstruct(self, loader):
        self.decoder.eval()
        corpus_lists = ()
        recon_lists = []
        target_lists = []
        with torch.no_grad():
            for batch, (corpus, emb, target) in enumerate(loader):
                emb, target = emb.to(self.device), target.to(self.device)
                decoded = self.decoder(emb)
                
                corpus_lists = corpus_lists + tuple(corpus)
                recon_lists.append(decoded.reshape(decoded.shape[0], -1))
                target_lists.append(target.reshape(target.shape[0], -1))

        return torch.cat(recon_lists, dim=0).cpu().detach().numpy(), torch.cat(target_lists, dim=0).cpu().detach().numpy(), corpus_lists 
        

    def visualize(self, validation_set, validation_loader):
        # Pre-Define Document to check
        doc_idx = []
        for idx in range(100):
            doc_idx.append(random.randint(0, len(validation_set)-1))

        # visualize documents
        for idx in doc_idx:
            # get recontruct result
            recon_list, target_list, doc_list = self.get_reconstruct(validation_loader)
            # get ranking index
            recon_rank_list = np.zeros((len(recon_list), len(self.vocabulary)), dtype='float32')
            target_rank_list = np.zeros((len(recon_list), len(self.vocabulary)), dtype='float32')
            for i in range(len(recon_list)):
                recon_rank_list[i] = np.argsort(recon_list[i])[::-1]
                target_rank_list[i] = np.argsort(target_list[i])[::-1]

            # show info
            record = open('./'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['target']+'_document.txt', 'a')
            print('Documents ', idx)
            record.write('Documents '+str(idx)+'\n')
            print(doc_list[idx])
            record.write(doc_list[idx])
            print('---------------------------------------')
            record.write('\n---------------------------------------\n')
            print('[Predict] Top 10 Words in Document: ')
            record.write('[Predict] Top 10 Words in Document: \n')
            for word_idx in range(10):
                print(self.id2token[recon_rank_list[idx][word_idx]])
                record.write(str(self.id2token[recon_rank_list[idx][word_idx]])+'\n')
            print('---------------------------------------')
            record.write('---------------------------------------\n')
            print('[Label] Top 10 Words in Document: ')
            record.write('[Label] Top 10 Words in Document: \n')
            for idx in range(10):
                print(self.id2token[target_rank_list[idx][idx]])
                record.write(str(self.id2token[target_rank_list[idx][idx]])+'\n')
            print('---------------------------------------\n')
            record.write('---------------------------------------\n\n')