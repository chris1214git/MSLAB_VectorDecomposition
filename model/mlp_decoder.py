import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from utils.loss import ListNet, MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings

class MLPDataset(Dataset):
    def __init__(self, emb, target):
        
        assert len(emb) == len(target)
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)        
        
    def __getitem__(self, idx):
        return self.emb[idx], self.target[idx]

    def __len__(self):
        return len(self.emb)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLPNetwork, self).__init__()
        self.output_dim = output_dim
       
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(input_dim*4),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(output_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        decoded = self.network(x)
        return decoded

class MLP:
    def __init__(self, config, vocabulary=None, contextual_size=768, word_embeddings=None):
        if torch.cuda.is_available():
            self.device = get_free_gpu()
        else:
            self.device = torch.device("cpu")
        self.config = config
        self.vocabulary = vocabulary
        self.contextual_size = contextual_size
        self.num_epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.word_embeddings = word_embeddings
        self.decoder = MLPNetwork(input_dim=contextual_size, output_dim=len(vocabulary), hidden_dim=300)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        if config['loss'] == 'bce':
            self.loss_funct = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif config['loss'] == 'mse':
            self.loss_funct = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_funct = MythNet()

    def fit(self, training_set, validation_set):
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.decoder = self.decoder.to(self.device)

        for epoch in range(self.num_epochs):
            self.decoder.train()
            for batch, (emb, target) in enumerate(tqdm(training_loader, desc="Training")):
                emb, target = emb.to(self.device), target.to(self.device)
                loss = self.loss_funct(self.decoder(emb), target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if  (epoch + 1) % 10 == 0:
                validation_result = self.validation(validation_loader)
                if self.config['dataset2'] is not None:
                    record = open('./'+'CrossDomain_'+self.config['dataset']+'_'+self.config['dataset2']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['loss']+'_'+self.config['target']+'.txt', 'a')
                else:
                    record = open('./'+'baseline_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['loss']+'_'+self.config['target']+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print('EPOCH', epoch + 1)
                record.write('EPOCH '+ str(epoch + 1) + '\n')
                for key,val in validation_result.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")

    def validation(self, loader):
        self.decoder.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (emb, target) in enumerate(loader):
                emb, target = emb.to(self.device), target.to(self.device)
                recon_dists = self.decoder(emb)
                # Semantic Prcision for reconstruct
                precision_scores, word_result = semantic_precision_all(recon_dists, target, self.word_embeddings, self.vocabulary, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    results['[Recon] Semantic Precision v1@{}'.format(k)].append(v)
                precision_scores, word_result = semantic_precision_all_v2(recon_dists, target, self.word_embeddings, self.vocabulary, k=self.config['topk'], th = self.config['threshold'])
                for k, v in precision_scores.items():
                    results['[Recon] Semantic Precision v2@{}'.format(k)].append(v)
                    
                # Precision for reconstruct
                precision_scores = retrieval_precision_all(recon_dists, target, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v1@{}'.format(k)].append(v)
                precision_scores = retrieval_precision_all_v2(recon_dists, target, k=self.config['topk'])
                for k, v in precision_scores.items():
                    results['[Recon] Precision v2@{}'.format(k)].append(v)

                # NDCG for reconstruct
                ndcg_scores = retrieval_normalized_dcg_all(recon_dists, target, k=self.config['topk'])
                for k, v in ndcg_scores.items():
                    results['[Recon] ndcg@{}'.format(k)].append(v)
            for k in results:
                results[k] = np.mean(results[k])
        return results
