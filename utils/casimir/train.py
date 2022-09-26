import math
import os
import re
from collections import defaultdict

import argparse
import numpy as np
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from utils.Loss import ListNet, topk_ListNet
from utils.data_preprocessing import get_process_data

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class DocDataset(Dataset):
  def __init__(self, vocab_size, document_tfidf, documnet_weight, document_tfidf_rank, document_class):
    self.vocab_size = vocab_size
    self.document_tfidf = torch.nn.functional.normalize(torch.FloatTensor(document_tfidf))
    self.documnet_weight = torch.FloatTensor(documnet_weight)
    self.document_tfidf_rank = torch.LongTensor(document_tfidf_rank)
    self.document_class = torch.LongTensor(document_class)
  
  def __getitem__(self, idx):
    return self.document_tfidf[idx], self.documnet_weight[idx], self.document_tfidf_rank[idx], self.document_class[idx]
  
  def __len__(self):
    return len(self.document_tfidf)

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.Tanh(),
            nn.Linear(4096, 1024),
            nn.Tanh(),
            nn.Linear(1024, output_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)

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

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 20),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)

def evaluate_sklearn(pred, ans):
    results = {}
        
    one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    
    for topk in [10, 30, 50]:
        one_hot_pred = np.argsort(pred)[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        recall = len(hit) / len(one_hot_ans)
        f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
        
        results['F1@{}'.format(topk)] = f1
        
    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in [10, 30, 50]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
    return results



main_path = './'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_AE')
parser.add_argument('--type', type=str, default='50')
arg = parser.parse_args()

nltk.download('stopwords')
same_seeds(123)

data_dict = get_process_data("20news", word2embedding_path=main_path+'glove.6B.100d.txt')
doc_tfidf = data_dict['document_tfidf']
doc_weight = data_dict['document_weight']
doc_error = np.intc(data_dict['document_error'])
doc_target = np.delete(data_dict['dataset']['target'], doc_error, 0)
num_classes = data_dict['dataset']['num_classes']

train_doc_num = len(doc_tfidf)
vocab_size = len(doc_tfidf[0])
doc_tfidf_rank = np.zeros((train_doc_num, vocab_size), dtype='float32')
    
for i in tqdm(range(train_doc_num)):
    doc_tfidf_rank[i] = np.argsort(doc_tfidf[i])[::-1]

train_size_ratio = 0.8
train_size = int(train_doc_num * train_size_ratio)
epochs = 150
batch_size = 128
lr = 0.001
    
dataset = DocDataset(vocab_size, doc_tfidf, doc_weight, doc_tfidf_rank, doc_target)
training_set, validation_set = random_split(dataset, lengths=[train_size, train_doc_num - train_size])
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=False)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=False)

if arg.mode == 'train_AE':
    embedding_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {} for training".format(device))

    model_encoder = Encoder(vocab_size, embedding_dim).to(device)
    model_decoder = Decoder(embedding_dim, vocab_size).to(device)
    optimizer_en = torch.optim.Adam(model_encoder.parameters(), lr=lr)
    optimizer_de = torch.optim.Adam(model_decoder.parameters(), lr=lr)

    log_train_loss = []
    log_val_loss = []

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        model_encoder.train()
        model_decoder.train()
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            data_tfidf, data_rank, label = data_tfidf.to(device), data_rank.to(device), label.to(device)
            encoded = model_encoder(data_tfidf)
            decoded = model_decoder(encoded)
            decoded = torch.nn.functional.normalize(decoded, dim=1)
            loss = topk_ListNet(decoded, data_tfidf,data_rank, topk=64)
                
            optimizer_en.zero_grad()
            optimizer_de.zero_grad()

            loss.backward()
            optimizer_en.step()
            optimizer_de.step()
            total_train_loss += loss

        model_encoder.eval()
        model_decoder.eval()
        results = []
        with torch.no_grad():
            for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(validation_loader)):
                data_tfidf, data_rank, label = data_tfidf.to(device), data_rank.to(device), label.to(device)
                coded = model_encoder(data_tfidf)
                decoded = model_decoder(coded)
                loss = topk_ListNet(decoded, data_tfidf,data_rank, topk=64)
                total_val_loss += loss
                if (epoch + 1) % 10 == 0:
                    decoded = decoded.cpu()
                    data_tfidf = data_tfidf.cpu()
                    for idx in range(len(data_tfidf)):
                        res = evaluate_sklearn(decoded[idx], data_tfidf[idx])
                        results.append(res)
        if (epoch + 1) % 10 == 0:
            results_m = pd.DataFrame(results).mean()
            print('------'+str(epoch)+'------')
            print(results_m)
            print('-------------------------------')
        total_train_loss /= len(training_loader)
        total_val_loss /= len(validation_loader)
        log_train_loss.append(total_train_loss)
        log_val_loss.append(total_val_loss)
        print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))
        if((epoch+1) % 10 == 0):
            torch.save(model_encoder.state_dict(), main_path+'model/'+str(epoch+1)+'_en.pth')
            torch.save(model_decoder.state_dict(), main_path+'model/'+str(epoch+1)+'_de.pth')

    print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))
    
elif arg.mode =='train_DE':
    embedding_dim = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {} for training".format(device))

    model_decoder = Decoder(embedding_dim, vocab_size).to(device)
    optimizer_de = torch.optim.Adam(model_decoder.parameters(), lr=lr)

    log_train_loss = []
    log_val_loss = []

    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        model_decoder.train()
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            data_tfidf, data_weight, data_rank, label = data_tfidf.to(device), data_weight.to(device), data_rank.to(device), label.to(device)
            decoded = model_decoder(data_weight)
            decoded = torch.nn.functional.normalize(decoded, dim=1)
            loss = topk_ListNet(decoded, data_tfidf,data_rank, topk=64)
                
            optimizer_de.zero_grad()

            loss.backward()
            optimizer_de.step()
            total_train_loss += loss

        model_decoder.eval()
        results = []
        with torch.no_grad():
            for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(validation_loader)):
                data_tfidf, data_weight, data_rank, label = data_tfidf.to(device), data_weight.to(device), data_rank.to(device), label.to(device)
                decoded = model_decoder(data_weight)
                loss = topk_ListNet(decoded, data_tfidf,data_rank, topk=64)
                total_val_loss += loss
                if (epoch + 1) % 10 == 0:
                    decoded = decoded.cpu()
                    data_tfidf = data_tfidf.cpu()
                    for idx in range(len(data_tfidf)):
                        res = evaluate_sklearn(decoded[idx], data_tfidf[idx])
                        results.append(res)
        if (epoch + 1) % 10 == 0:
            results_m = pd.DataFrame(results).mean()
            print('------'+str(epoch)+'------')
            print(results_m)
            print('-------------------------------')
        total_train_loss /= len(training_loader)
        total_val_loss /= len(validation_loader)
        log_train_loss.append(total_train_loss)
        log_val_loss.append(total_val_loss)
        print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))
        # if((epoch+1) % 10 == 0):
            # torch.save(model_decoder.state_dict(), main_path+'model/'+str(epoch+1)+'_de.pth')

    print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))

elif arg.mode == "downstream_AE_MLP":
    epochs = 50
    embedding_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {} for training".format(device))

    model_encoder = Encoder(vocab_size, embedding_dim).to(device)
    model_encoder.load_state_dict(torch.load(main_path+'model/'+arg.type+'_en.pth'))
    model_classifier = Classifier(embedding_dim, num_classes).to(device)
    optimizer_class = torch.optim.Adam(model_classifier.parameters(), lr = lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_train_loss = 0
        train_acc = []
        total_val_loss = 0
        val_acc = []
        model_encoder.eval()
        model_classifier.train()
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            data_tfidf, label = data_tfidf.to(device), label.to(device)
            with torch.no_grad():
                encoded = model_encoder(data_tfidf)
            pred = model_classifier(encoded)
                
            loss = loss_function(pred, label)
            total_train_loss += loss
                
            hits = pred.argmax(dim=1).eq(label)
            train_acc.append(hits)

            optimizer_class.zero_grad()
            loss.backward()
            optimizer_class.step()

        model_classifier.eval()
        with torch.no_grad():
            for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(validation_loader)):
                data_tfidf, label = data_tfidf.to(device), label.to(device)
                encoded = model_encoder(data_tfidf)
                pred = model_classifier(encoded)

                loss = loss_function(pred, label)
                total_val_loss += loss

                hits = pred.argmax(dim=1).eq(label)
                val_acc.append(hits)
        total_train_loss /= len(training_loader)
        total_val_loss /= len(validation_loader)
        print("[{}/{}] Loss: {} ACC: {}/ Val Loss: {} Val ACC: {}".format(epoch+1, epochs, total_train_loss.item(), torch.cat(train_acc).float().mean(), total_val_loss.item(), torch.cat(val_acc).float().mean()))
    torch.save(model_classifier.state_dict(), main_path+'model/ae_classifier.pth')
    
elif arg.mode == "downstream_AE_LR":
    embedding_dim = 128
    model_encoder = Encoder(vocab_size, embedding_dim)
    model_encoder.load_state_dict(torch.load(main_path+'model/'+arg.type+'_en.pth'))
    model_encoder.eval()
    weight_vec = []
    label_vec = []
    with torch.no_grad():
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            weight_vec.append(model_encoder(data_tfidf))
            label_vec.append(label)
    weight_np = torch.cat(weight_vec, dim=0).detach().cpu().numpy()
    label_np = torch.cat(label_vec, dim=0).detach().cpu().numpy()
    ###
    model_LR = LogisticRegression(dual=False, solver='lbfgs', max_iter=5000)
    model_LR.fit(weight_np, label_np)
    # model_SVM = make_pipeline(StandardScaler(), LinearSVC(dual=False, tol=1e-5, max_iter=5000))
    # model_SVM.fit(weight_np, label_np)
    ###

    score = []
    with torch.no_grad():
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(validation_loader)):
            encoded = model_encoder(data_tfidf)
            ###
            score.append(model_LR.score(encoded, label))
            # score.append(model_SVM.score(encoded, label))
            ###
        print(np.mean(score))
        
elif arg.mode == "downstream_DE_MLP":
    epochs = 50
    embedding_dim = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {} for training".format(device))

    model_classifier = Classifier(embedding_dim, num_classes).to(device)
    optimizer_class = torch.optim.Adam(model_classifier.parameters(), lr = lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_train_loss = 0
        train_acc = []
        total_val_loss = 0
        val_acc = []
        model_classifier.train()
        for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            data_weight, label = data_weight.to(device), label.to(device)
            pred = model_classifier(data_weight)
                
            loss = loss_function(pred, label)
            total_train_loss += loss
                
            hits = pred.argmax(dim=1).eq(label)
            train_acc.append(hits)

            optimizer_class.zero_grad()
            loss.backward()
            optimizer_class.step()

        model_classifier.eval()
        with torch.no_grad():
            for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(validation_loader)):
                data_weight, label = data_weight.to(device), label.to(device)
                pred = model_classifier(data_weight)

                loss = loss_function(pred, label)
                total_val_loss += loss

                hits = pred.argmax(dim=1).eq(label)
                val_acc.append(hits)
        total_train_loss /= len(training_loader)
        total_val_loss /= len(validation_loader)
        print("[{}/{}] Loss: {} ACC: {}/ Val Loss: {} Val ACC: {}".format(epoch+1, epochs, total_train_loss.item(), torch.cat(train_acc).float().mean(), total_val_loss.item(), torch.cat(val_acc).float().mean()))
    torch.save(model_classifier.state_dict(), main_path+'model/de_classifier.pth')

elif arg.mode == "downstream_DE_LR":
    weight_vec = []
    label_vec = []
    for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
            weight_vec.append(data_weight)
            label_vec.append(label)
    weight_np = torch.cat(weight_vec, dim=0).detach().cpu().numpy()
    label_np = torch.cat(label_vec, dim=0).detach().cpu().numpy()
    ###
    model_LR = LogisticRegression(dual=False, solver='lbfgs', max_iter=5000)
    model_LR.fit(weight_np, label_np)
    #model_SVM = make_pipeline(StandardScaler(), LinearSVC(dual=False, tol=1e-5, max_iter=5000))
    #model_SVM.fit(weight_np, label_np)
    ###
    score = []
    for batch, (data_tfidf, data_weight, data_rank, label) in enumerate(tqdm(training_loader)):
        ###
        score.append(model_LR.score(data_weight, label))
        #score.append(model_SVM.score(data_weight, label))
        ###
    print(np.mean(score))

