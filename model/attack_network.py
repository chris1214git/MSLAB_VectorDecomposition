from array import array
from cProfile import label
import enum
from json import load
import random
from tkinter.tix import Tree
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from transformers import AdamW, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from tqdm.auto import tqdm

from utils.loss import ListNet, MythNet
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import get_free_gpu, record_settings

class AttackDataset(Dataset):
    def __init__(self, documents, emb, target):
        # Attack require raw document to train surrogate model
        assert len(emb) == len(target) and len(target) == len(documents)
        self.documents = documents
        self.emb = torch.FloatTensor(emb)
        self.target = torch.FloatTensor(target)
        
    def __getitem__(self, idx):
        return self.documents[idx], self.emb[idx], self.target[idx]

    def __len__(self):
        return len(self.emb)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.BatchNorm1d(input_dim*4),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim*4, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.network(x)

class Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Surrogate model: Transform raw document to doc embedding
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

    def forward(self, documents):
        return self.get_docvec(documents)

    def get_docvec(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to(self.device)
        embedding = self.model.bert(**inputs).last_hidden_state[:, 0, :]
        return embedding

class AttackNetwork():
    def __init__(self, config, vocabulary, contextual_size, word_embeddings) -> None:
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

        # Attack modules contain surrogate model and decoder
        self.encoder = Encoder(self.device)
        self.decoder = MLPNetwork(input_dim=contextual_size, output_dim=len(vocabulary))
    
    def train_surrogate(self, loader):
        # Training to make surrogate model act like black box encoder
        self.encoder.train()
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr, eps=1e-8)
        loss_func = torch.nn.MSELoss()
        for epoch in range(5):
            train_loss = 0
            for batch, (corpus, embs, labels) in enumerate(loader):
                embs = embs.to(self.device)     # [batch_size, embedding_dim]
                predict = self.encoder(corpus)  # [batch_size, embedding_dim]
                loss = loss_func(embs, predict)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print("")
            print("Epoch[{}/{}] Encoder loss:{}\n".format(epoch+1, self.num_epochs, train_loss / len(loader)))

    def generate_fake_data(self, train_loader, test_loader):
        self.encoder.eval()
        documents = []
        doc_embs = []
        label_list = []
        weights = []
        for batch, (corpus, embs, labels) in enumerate(train_loader):

            for document, emb, label in zip(corpus, embs, labels):
                documents.append(document)
                doc_embs.append(emb.numpy())
                label_list.append(label.numpy())
                weights.append(len(test_loader))

        for batch, (corpus, embs, labels) in enumerate(test_loader):
            fake_embs = self.encoder(corpus).detach().cpu()

            for document, fake_emb, label in zip(corpus, fake_embs, labels):
                documents.append(document)
                doc_embs.append(fake_emb.numpy())
                label_list.append(label.numpy())
                weights.append(len(train_loader))

        dataset = AttackDataset(documents, doc_embs, label_list)

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, sampler=sampler)

    def validation(self, loader):
        self.decoder.eval()
        results = defaultdict(list)
        with torch.no_grad():
            for batch, (_, emb, target) in enumerate(loader):
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

    def train_decoder(self, training_loader, validation_loader):
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        if self.config['loss'] == 'bce':
            self.loss_funct = torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif self.config['loss'] == 'mse':
            self.loss_funct = torch.nn.MSELoss(reduction='mean')
        elif self.config['loss'] == 'listnet':
            self.loss_funct = MythNet
        else:
            raise NotImplementedError("{} loss is not a valid loss function".format(self.config['loss']))

        for epoch in range(self.num_epochs):
            self.decoder.train()
            for batch, (_, emb, target) in enumerate(tqdm(training_loader, desc="Training")):
                emb, target = emb.to(self.device), target.to(self.device)
                loss = self.loss_funct(self.decoder(emb), target)
                loss.backward()
                self.decoder_optimizer.step()
                self.decoder_optimizer.zero_grad()
            if  (epoch + 1) % 10 == 0:
                validation_result = self.validation(validation_loader)
                record = open('./'+'Attack_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['loss']+'_'+self.config['target']+'.txt', 'a')
                print('---------------------------------------')
                record.write('-------------------------------------------------\n')
                print('EPOCH', epoch + 1)
                record.write('EPOCH '+ str(epoch + 1) + '\n')
                for key,val in validation_result.items():
                    print(f"{key}:{val:.4f}")
                    record.write(f"{key}:{val:.4f}\n")

    def fit(self, preprocessed_corpus, doc_embs, label):
        # prepare dataset
        dataset = AttackDataset(preprocessed_corpus, doc_embs, label)
        training_length = int(len(dataset) * self.config['ratio'])
        validation_length = len(dataset) - training_length
        training_set, validation_set = random_split(dataset, lengths=[training_length, validation_length],generator=torch.Generator().manual_seed(42))

        if self.config["inductive"]:
            # Inductive
            val_length = int(validation_length * 0.8)
            test_length = validation_length - val_length
            validation_set, testing_set = random_split(validation_set, lengths=[val_length, test_length])
        else:
            # Transductive
            testing_set = validation_set

        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        testing_loader = DataLoader(testing_set, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # First stage, train surrogate mode
        self.train_surrogate(training_loader)

        # Second stage, generate training pairs and train decoder
        fake_loader = self.generate_fake_data(training_loader, validation_loader)
        self.train_decoder(fake_loader, testing_loader)

        # Third stage, test the result
        testing_result = self.validation(testing_loader)
        record = open('./'+'Attack_'+self.config['dataset']+'_'+self.config['model']+'_'+self.config['encoder']+'_'+self.config['loss']+'_'+self.config['target']+'.txt', 'a')
        print('---------------------------------------')
        record.write('-------------------------------------------------\n')
        print("Final testing result")
        record.write("Final testing result")
        for key,val in testing_result.items():
            print(f"{key}:{val:.4f}")
            record.write(f"{key}:{val:.4f}\n")