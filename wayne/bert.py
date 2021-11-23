import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

sys.path.append("../..")

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import ndcg_score

from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
from sklearn.linear_model import LogisticRegression
from utils.Loss import ListNet
from utils.data_processing import get_process_data

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_sklearn(pred, ans, config):
    results = {}

    one_hot_ans = np.arange(ans.shape[0])[ans > 0]

    for topk in config["topk"]:
        one_hot_pred = np.argsort(pred)[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        # print(percision)
        recall = len(hit) / len(one_hot_ans)
        # print(recall)
        f1 = 2 * percision * recall / \
            (percision + recall) if (percision + recall) > 0 else 0

        results['F1@{}'.format(topk)] = f1

    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))

    return results

def generate_document_embedding(model, data_loader):
    model.eval()
    document_vectors = []

    with torch.no_grad():
        for documents, targets, labels, ranks in tqdm(data_loader):
            vec = model.get_docvec(documents)
            document_vectors.append(vec)

    document_vectors = torch.cat(
        document_vectors, dim=0).detach().cpu().numpy()
    return document_vectors


def evaluate_downstream(document_vectors, targets, train_dataset, test_dataset):
    # LR 
    train_x = document_vectors[train_dataset.indices,:]
    train_y = targets[train_dataset.indices]

    test_x = document_vectors[test_dataset.indices,:]
    test_y = targets[test_dataset.indices]
    
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(train_x, train_y)
    preds = classifier.predict(test_x)
    valid_acc = np.mean(preds == test_y)
    
    return valid_acc

class BertDataset(Dataset):

    def __init__(self, documents, targets, labels, ranks) -> None:
        super().__init__()
        self.documents = documents
        self.targets = torch.LongTensor(targets)
        self.labels = torch.tensor(labels)
        self.ranks = torch.LongTensor(ranks)
        assert len(documents) == len(targets) and len(documents) == len(labels)

    def __len__(self):
        return len(documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.targets[idx], self.labels[idx], self.ranks[idx]


class BertFamily(nn.Module):
    def __init__(self, member, device,):
        super().__init__()
        self.member = member
        self.device = device
        if member == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        elif member =='albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = AlbertForMaskedLM.from_pretrained('albert-base-v2').to(device)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

    def forward(self, documents):
        return self.get_docvec(documents)

    def generate_mask_input(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', 
                                max_length=128, truncation=True, padding=True)
        inputs['labels'] = inputs.input_ids.detach().clone()
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        
        return inputs

    def mlm_pretrain(self, documents):

        inputs = self.generate_mask_input(documents).to(self.device)

        outputs = self.model(**inputs)

        return outputs.loss

    def get_docvec(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to(self.device)
        if self.member == 'roberta':
            embedding = self.model.roberta(**inputs).last_hidden_state[:, 0, :]
        elif self.member == 'albert':
            embedding = self.model.albert(**inputs).last_hidden_state[:, 0, :]
        else:
            embedding = self.model.bert(**inputs).last_hidden_state[:, 0, :]
        return embedding

class Decoder(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.decoder = nn.Sequential(
        nn.Linear(input_dim, 1024),
        # nn.Dropout(p=0.5),
        nn.Tanh(),
        nn.Linear(1024, 4096),
        # nn.Dropout(p=0.5),
        nn.Tanh(),
        nn.Linear(4096, output_dim),
        # nn.Dropout(p=0.5),
        nn.Sigmoid(),
    )
  
  def forward(self, x):
    return self.decoder(x)

def train_model(config, data_loader, training_set, validation_set, output_dim, targets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(config['gpu'])
    embedding_dim = 768
    
    train_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    encoder = BertFamily(config['member'], device)
    decoder = Decoder(embedding_dim, output_dim).to(device)
    if config['scheduler'] == 'required':
        total_steps = len(train_loader) * config['epochs']
        optimizer_en = AdamW(encoder.parameters(), lr=config['lr'], eps=1e-8)
        optimizer_de = AdamW(decoder.parameters(), lr=config['lr'], eps=1e-8)
        scheduler_en = get_linear_schedule_with_warmup(optimizer_en, num_warmup_steps=0, num_training_steps=total_steps)
        scheduler_de = get_linear_schedule_with_warmup(optimizer_de, num_warmup_steps=0, num_training_steps=total_steps)
    else:
        optimizer_en = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
        optimizer_de = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

    print('-------- Info ---------')
    print('Bert Family: {}\nDataset: {}\nEpochs: {}\nBatch Size: {}\nLearning Rate: {}\nScheduler: {}\nMaskLM pretrain: {}\nDevice: {}\nDevice Num: {}'.format(config['member'], config['dataset'], config['epochs'], config['batch_size'], config['lr'], config['scheduler'], config['mlm_pretrain'], device, config['gpu']))
    print('-----------------------')

    for epoch in range(config['epochs']):
        # Training
        encoder.train()
        decoder.train()
        train_loss = 0
        for idx, (doc, target, label, rank) in enumerate(tqdm(train_loader, desc="Training")):
            embedding = encoder.get_docvec(doc).to(device)
            label = torch.nn.functional.normalize(label.to(device), dim=1)
            decoded = torch.nn.functional.normalize(decoder(embedding), dim=1)
            decode_loss = ListNet(decoded, label)
            if config["mlm_pretrain"] == "True":
                mlm_loss = encoder.mlm_pretrain(doc)
                loss = decode_loss + mlm_loss
            else:
                loss = decode_loss
            
            loss.backward()
            optimizer_en.step()
            optimizer_de.step()
            if config['scheduler'] == 'required':
                scheduler_en.step()
                scheduler_de.step()
            optimizer_en.zero_grad()
            optimizer_de.zero_grad()

            train_loss += loss.item()
        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0
        results = []
        with torch.no_grad():
            for batch, (doc, target, label, rank) in enumerate(tqdm(validation_loader, desc='Validation')):
                embedding = encoder.get_docvec(doc).to(device)
                label = torch.nn.functional.normalize(label.to(device), dim=1)
                decoded = torch.nn.functional.normalize(decoder(embedding), dim=1)
                decode_loss = ListNet(decoded, label)
                if config["mlm_pretrain"] == "True":
                    mlm_loss = encoder.mlm_pretrain(doc)
                    loss = decode_loss + mlm_loss
                else:
                    loss = decode_loss
                val_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    decoded = decoded.cpu()
                    label = label.cpu()
                    for idx in range(len(label)):
                        res = evaluate_sklearn(decoded[idx], label[idx], config)
                        results.append(res)
        # F1 Score & NDCG & Downstream ACC
        if (epoch + 1) % 10 == 0:
            doc_emb = generate_document_embedding(encoder, data_loader)
            val_acc = evaluate_downstream(doc_emb, targets, training_set, validation_set) 
            record = open('./'+config['member']+'_mlm'+config['mlm_pretrain']+'_sheduler'+config['scheduler']+'.txt', 'a')
            results_m = pd.DataFrame(results).mean()
            record.write('------'+str(epoch)+'------')
            record.write(str(results_m))
            record.write('ACC: '+str(val_acc))
            record.write('-------------------------------')
            record.close()
            print('------'+str(epoch)+'------')
            print(results_m)
            print('ACC: '+str(val_acc))
            print('-------------------------------')
        print("[{}/{}] Training Loss: {} / Validation Loss: {}".format(epoch +
              1, config['epochs'], train_loss/len(train_loader), val_loss/len(validation_loader)))

    return encoder, decoder

def train_encoder(config, data_loader, targets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(config['gpu'])

    encoder = BertFamily(config['member'], device)
    if config['scheduler'] == 'required':
        total_steps = len(data_loader) * config['epochs']
        optimizer_en = AdamW(encoder.parameters(), lr=config['lr'], eps=1e-8)
        scheduler_en = get_linear_schedule_with_warmup(optimizer_en, num_warmup_steps=0, num_training_steps=total_steps)
    else:
        optimizer_en = torch.optim.Adam(encoder.parameters(), lr=config['lr'])

    print('-------- Info ---------')
    print('Bert Family: {}\nDataset: {}\nEpochs: {}\nBatch Size: {}\nLearning Rate: {}\nScheduler: {}\nMaskLM pretrain: {}\nDevice: {}\nDevice Num: {}'.format(config['member'], config['dataset'], config['epochs'], config['batch_size'], config['lr'], config['scheduler'], config['mlm_pretrain'], device, config['gpu']))
    print('-----------------------')

    for epoch in range(config['epochs']):
        # Training
        encoder.train()
        train_loss = 0
        for idx, (doc, target, label, rank) in enumerate(tqdm(data_loader, desc="Training")):
            mlm_loss = encoder.mlm_pretrain(doc)
            loss = mlm_loss

            loss.backward()
            optimizer_en.step()
            if config['scheduler'] == 'required':
                scheduler_en.step()
            optimizer_en.zero_grad()

            train_loss += loss.item()
        # F1 Score & NDCG & Downstream ACC
        if (epoch + 1) % 10 == 0:
            doc_emb = generate_document_embedding(encoder, data_loader)
            val_acc = evaluate_downstream(doc_emb, targets, training_set, validation_set) 
            record = open('./'+config['member']+'_mlm'+config['mlm_pretrain']+'_sheduler'+config['scheduler']+'.txt', 'a')
            record.write('------'+str(epoch)+'------')
            record.write('ACC: '+str(val_acc))
            record.write('-------------------------------')
            record.close()
            print('------'+str(epoch)+'------')
            print('ACC: '+str(val_acc))
            print('-------------------------------')
        print("[{}/{}] Training Loss: {}".format(epoch+1, config['epochs'], train_loss/len(data_loader)))

    return encoder
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--label', type=str, default="tfidf")
    parser.add_argument('--member', type=str, default='bert')       # (1) bert (2) roberta (3) albert
    parser.add_argument('--mlm_pretrain', type=str, default="True") # (1) True: ListNet + MLM (2) False: ListNet (3) Only: MLM
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='False')
    parser.add_argument('--seed', type=int, default=123)   
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
    args = parser.parse_args()
    config = vars(args)

    same_seeds(config["seed"])

    data_dict = get_process_data(config["dataset"])
    raw_data = data_dict["dataset"]
    documents, targets, target_num = raw_data["documents"], raw_data["target"], raw_data["num_classes"]

    if (config["label"] == "tfidf"):
        labels = data_dict["document_tfidf"]
    else:
        labels = None

    doc_num = len(labels)
    vocab_size = len(labels[0])
    ranks = np.zeros((doc_num, vocab_size), dtype='float32')
    for i in range(doc_num):
        ranks[i] = np.argsort(labels[i])[::-1]

    train_size_ratio = 0.8
    train_size = int(doc_num * train_size_ratio)

    dataset = BertDataset(documents, targets, labels, ranks)
    data_loader = DataLoader(dataset,  batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    training_set, validation_set = random_split(dataset, lengths=[train_size, doc_num-train_size], generator=torch.Generator().manual_seed(42))
    
    if config['mlm_pretrain'] == 'Only':
        encoder = train_encoder(config, data_loader, targets)
    else:
        encoder, decoder = train_model(config, data_loader, training_set, validation_set, vocab_size, targets)

