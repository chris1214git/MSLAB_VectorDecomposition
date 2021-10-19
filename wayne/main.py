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

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_document(dataset):
    if dataset == "20news":
        num_classes = 20
        raw_text, target = fetch_20newsgroups(data_home="./", subset='all', categories=None,
                                shuffle=True, random_state=42, return_X_y=True)
        documents = [doc.strip("\n") for doc in raw_text]
    elif dataset == "IMDB":
        num_classes = 2
        documents = []
        target = []

        dir_prefix = "./aclImdb/train/"
        sub_dir = ["pos","neg"]
        for target_type in sub_dir:
            data_dir = os.path.join(dir_prefix, target_type)
            files_name = os.listdir(data_dir)
            for f_name in files_name:
                with open(os.path.join(data_dir,f_name),"r") as f:
                    context = f.readlines()
                    documents.extend(context)

            # assign label
            label = 1 if target_type=="pos" else 0
            label = [label]* len(files_name)
            target.extend(label)
    else:
        raise NotImplementedError
    
    return documents, target, num_classes

class Vocabulary:
    def __init__(self, min_word_freq_threshold=0, topk_word_freq_threshold=0):
        # The low frequency words will be assigned as <UNK> token
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}
        
        self.min_word_freq_threshold = min_word_freq_threshold
        self.topk_word_freq_threshold = topk_word_freq_threshold
        
        self.word_freq_in_corpus = defaultdict(int)
        self.IDF = {}
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.strip().split()
        
        return [self.ps.stem(w) for w in text if w.lower() not in self.stop_words]

    def build_vocabulary(self, sentence_list):
        self.word_vectors = []
        self.doc_freq = defaultdict(int) # # of document a word appear
        self.document_num = len(sentence_list)
        
        for sentence in tqdm(sentence_list, desc="Preprocessing documents"):
            # for doc_freq
            document_words = set()
            
            for word in self.tokenizer_eng(sentence):
                # calculate word freq
                self.word_freq_in_corpus[word] += 1
                document_words.add(word)
                
            for word in document_words:
                self.doc_freq[word] += 1
        
        # calculate IDF
        print('doc num', self.document_num)
        for word, freq in self.doc_freq.items():
            self.IDF[word] = math.log(self.document_num / (freq+1))
        
        # delete less freq words:
        delete_words = []
        for word, v in self.word_freq_in_corpus.items():
            if v < self.min_word_freq_threshold:
                delete_words.append(word)     
        for word in delete_words:
            del self.IDF[word]    
            del self.word_freq_in_corpus[word]
        
        # delete too freq words
        print('eliminate freq words')
        IDF = [(word, freq) for word, freq in self.IDF.items()]
        IDF.sort(key=lambda x: x[1])

        for i in range(self.topk_word_freq_threshold):
            word = IDF[i][0]
            del self.IDF[word]
            del self.word_freq_in_corpus[word]
        
        # construct word_vectors
        idx = 1
        for word in self.word_freq_in_corpus:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
            
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] for token in tokenized_text if token in self.stoi
        ]

class DocDataset(Dataset):
  def __init__(self, vocab_size, document_tfidf, document_tfidf_rank):
    self.vocab_size = vocab_size
    # self.documnet_embedding = torch.FloatTensor(documnet_embedding)
    self.document_tfidf = torch.nn.functional.normalize(torch.FloatTensor(document_tfidf))
    self.document_tfidf_rank = torch.LongTensor(document_tfidf_rank)
    #self.device = device
  
  def __getitem__(self, idx):
    return self.document_tfidf[idx], self.document_tfidf_rank[idx]
  
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
        nn.Tanh(),
        nn.Linear(1024, 4096),
        nn.Tanh(),
        nn.Linear(4096, output_dim),
        nn.Sigmoid(),
    )
  
  def forward(self, x):
    return self.decoder(x)

def ListNet(y_pred, y_true, eps=1e-10):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    # pred_smax = torch.nn.functional.softmax(y_pred, dim=1)
    # true_smax = torch.nn.functional.softmax(y_true, dim=1)

    pred = y_pred + eps
    pred_log = torch.log(pred)

    return torch.mean(torch.sum(-y_true * pred_log, dim=1))

def topk_ListNet(pred, label, eps=1e-10, topk=50):
    # pred = pred.clone()
    # label = label.clone()
    # _, rank = torch.sort(label, dim=1, descending=True)

    # pred_rank = []
    # label_rank = []
    # for i in range(len(pred)):
    #     pred_score = []
    #     label_score = []
    #     for j in range(topk):
    #         pred_score.append(pred[i, rank[i, j].item()].item())
    #         label_score.append(label[i, rank[i, j].item()].item())
    #     pred_rank.append(pred_score)
    #     label_rank.append(label_score)
  
    # pred_rank = torch.tensor(pred_rank, requires_grad=True)
    # label_rank = torch.tensor(label_rank, requires_grad=True)

    # pred_smax = pred_rank + eps

    # return torch.mean(torch.sum(-label_rank * torch.log(pred_smax), dim=1))
    pred_rank, rank_p = torch.sort(pred, dim=1, descending=True)
    label_rank, rank_l = torch.sort(label, dim=1, descending=True)
    pred_smax = pred_rank + eps

    return torch.mean(torch.sum(-label_rank * torch.log(pred_smax), dim=1))

def evaluate_sklearn(pred, ans):
    results = {}
        
    one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    
    for topk in [10, 30, 50]:
        one_hot_pred = np.argsort(pred)[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        #print(percision)
        recall = len(hit) / len(one_hot_ans)
        #print(recall)
        f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
        
        results['F1@{}'.format(topk)] = f1
        
    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in [10, 30, 50]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
    return results


if __name__ == "__main__":
    main_path = './'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--type', type=str, default='50')
    arg = parser.parse_args()
    # Create Vocabulary
    nltk.download('stopwords')
    same_seeds(123)
    documents, target, num_classes = load_document("20news")
    max_seq_length = 64
    vocab = Vocabulary(min_word_freq_threshold=5, 
            topk_word_freq_threshold=100)
    vocab.build_vocabulary(documents)
    print(f"Vocab size:{len(vocab)}")

    # Prepare Document TFIDF Label
    tokenize_data = []
    train_doc_num = vocab.document_num
    for sen_id, sen in enumerate(tqdm(documents, desc="Numericalizing")):
        numerical_output = vocab.numericalize(sen)[:max_seq_length]
            
        # some document becomes empty after filtering word
        if len(numerical_output) > 0:
            tokenize_data.append(torch.LongTensor(numerical_output))

    doc_tfidf = np.zeros((train_doc_num, len(vocab)), dtype='float32') # range(vocab.document_num)
    doc_tfidf_rank = np.zeros((train_doc_num, len(vocab)), dtype='float32') # range(vocab.document_num)
    print(doc_tfidf.shape)

    for i in tqdm(range(train_doc_num), desc="Doc TFIDF"): # range(vocab.document_num)
        for idx in tokenize_data[i]:
            doc_tfidf[i, idx] += vocab.IDF[vocab.itos[int(idx)]]
        doc_tfidf_rank[i] = sorted(range(len(vocab)), key = lambda k: doc_tfidf[i][k], reverse=True)
        doc_tfidf_rank[i][50] = -1
    # del tokenize_data
    # del doc_tfidf
    # np.save(main_path+'20news/docvec_20news_LSTM_128d_label.npy', doc_tfidf_rank)

    # Prepare Document Embedding Data
    # doc_embedding = np.load(main_path+'20news/docvec_20news_LSTM_128d.npy')
    #doc_tfidf_rank = np.load(main_path+'20news/docvec_20news_LSTM_128d_label.npy')
    #print(doc_embedding.shape[1])

    train_size_ratio = 0.8
    train_size = int(train_doc_num * train_size_ratio)
    epochs = 150
    batch_size = 128
    lr = 0.001
    if arg.mode == 'train':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    embedding_dim = 128
    if arg.mode == 'train':
        model_encoder = Encoder(len(vocab), embedding_dim).to(device)
        model_decoder = Decoder(embedding_dim, len(vocab)).to(device)
        optimizer_en = torch.optim.Adam(model_encoder.parameters(), lr=lr)
        optimizer_de = torch.optim.Adam(model_decoder.parameters(), lr=lr)
        # loss_function = nn.MultiLabelMarginLoss()

        training_set = DocDataset(len(vocab), doc_tfidf[:train_size], doc_tfidf_rank[:train_size])
        validation_set = DocDataset(len(vocab), doc_tfidf[train_size:train_doc_num], doc_tfidf_rank[train_size:train_doc_num])
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, pin_memory=False)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=False)

        print("Using device {} for training".format(device))

        log_train_loss = []
        log_val_loss = []

        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            model_encoder.train()
            model_decoder.train()
            for batch, (data_tfidf, label) in enumerate(tqdm(training_loader)):
                data_tfidf, label = data_tfidf.to(device), label.to(device)
                encoded = model_encoder(data_tfidf)
                decoded = model_decoder(encoded)
                decoded = torch.nn.functional.normalize(decoded, dim=1)
                
                loss = ListNet(decoded, data_tfidf)
                
                optimizer_en.zero_grad()
                optimizer_de.zero_grad()

                loss.backward()
                optimizer_en.step()
                optimizer_de.step()
                total_train_loss += loss
        
            model_decoder.eval()
            with torch.no_grad():
                for batch, (data_tfidf, label) in enumerate(tqdm(validation_loader)):
                    data_tfidf, label = data_tfidf.to(device), label.to(device)
                    coded = model_encoder(data_tfidf)
                    decoded = model_decoder(coded)
                    loss = ListNet(decoded, data_tfidf)
                    total_val_loss += loss

            total_train_loss /= len(training_loader)
            total_val_loss /= len(validation_loader)
            log_train_loss.append(total_train_loss)
            log_val_loss.append(total_val_loss)
            print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))
            if((epoch+1) % 10 == 0):
                torch.save(model_encoder.state_dict(), main_path+'model/'+str(epoch+1)+'_en.pth')
                torch.save(model_decoder.state_dict(), main_path+'model/'+str(epoch+1)+'_de.pth')

        print("[{}/{}] Loss: {} / Val Loss: {}".format(epoch+1, epochs, total_train_loss.item(), total_val_loss.item()))
        plt.figure(figsize=(10, 5))
        plt.title("Training & Validation")
        plt.plot(log_train_loss.cpu(), label="train")
        plt.plot(log_val_loss.cpu(), label="validation")
        plt.savefig(main_path+'log.png')
    
    elif arg.mode == 'test':
        testing_set = DocDataset(len(vocab), doc_tfidf[train_size:train_doc_num], doc_tfidf_rank[trina_size:train_doc_num])
        testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, pin_memory=False)
        # model_encoder = Encoder(len(vocab), embedding_dim).to(device)
        model_decoder = Decoder(embedding_dim, len(vocab)).to(device)
        # model_encoder.load_state_dict(torch.load(main_path+'model/'+arg.type+'_en.pth'))
        model_decoder.load_state_dict(torch.load(main_path+'model/'+arg.type+'_de.pth'))
        # model_encoder.eval()
        model_decoder.eval()
        with torch.no_grad():
            for batch, (data, data_tfidf, label) in enumerate(tqdm(testing_loader)):
                # coded = model_encoder(data)
                decoded = model_decoder(coded)
                for idx in range(len(data)):
                    res = evaluate_sklearn(decoded[idx], data[idx])
                    result.append(res)
        results_m = pd.DataFrame(results).mean()
        print('------'+arg.type+'------')
        print(results_m)
        print('-------------------------------')
    elif arg.mode == 'testall':
        for model_type in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
            testing_set = DocDataset(len(vocab), doc_tfidf[train_size:train_doc_num], doc_tfidf_rank[train_size:train_doc_num])
            testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, pin_memory=False)
            model_encoder = Encoder(len(vocab), embedding_dim).to(device)
            model_decoder = Decoder(embedding_dim, len(vocab)).to(device)
            model_encoder.load_state_dict(torch.load(main_path+'model/'+str(model_type)+'_en.pth'))
            model_decoder.load_state_dict(torch.load(main_path+'model/'+str(model_type)+'_de.pth'))
            model_encoder.eval()
            model_decoder.eval()
            results = []
            with torch.no_grad():
                for batch, (data_tfidf, label) in enumerate(tqdm(testing_loader)):
                    coded = model_encoder(data_tfidf)
                    decoded = model_decoder(coded)
                    for idx in range(len(data_tfidf)):
                        res = evaluate_sklearn(decoded[idx], data_tfidf[idx])
                        results.append(res)
            results_m = pd.DataFrame(results).mean()
            print('------'+str(model_type)+'------')
            print(results_m)
            print('-------------------------------')


