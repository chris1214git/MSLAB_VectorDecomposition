#!/usr/bin/env python
# coding: utf-8

# ### raw data
# * word embedding: glove
# * doc text: ./data/IMDB.txt

# ### dataset
# 1. IMDB
# 2. CNNNews
# 3. [PubMed](https://github.com/LIAAD/KeywordExtractor-Datasets/blob/master/datasets/PubMed.zip)

# ### preprocess
# 1. filter too frequent and less frequent words
# 2. stemming
# 3. document vector aggregation

# ### model
# 1. TopK
# 2. Sklearn
# 3. Our model

# ### evaluation
# 1. F1
# 2. NDCG

# In[1]:


import os
from collections import defaultdict
import math
import numpy as np 
import re
import torch
import torch.nn as nn
from itertools import cycle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm

# Used to get the data
from sklearn.metrics import ndcg_score

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
nltk.download('stopwords')

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# ## Preprocess config

# In[2]:

import argparse
parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="IMDB")
parser.add_argument('--n_document', type=int, default=500)
parser.add_argument('--normalize_word_embedding', action='store_true')
parser.add_argument('--min_word_freq_threshold', type=int, default=20)
parser.add_argument('--topk_word_freq_threshold', type=int, default=300)
parser.add_argument('--document_vector_agg_weight', type=str, default='mean')
parser.add_argument('--embedding_file', type=str, default='glove.6B.100d.txt')
parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])

args = parser.parse_args()
# In[2]:


config = {}

config["dataset"] = args.dataset # "IMDB" "CNN", "PubMed"
config["n_document"] = args.n_document
config["normalize_word_embedding"] = args.normalize_word_embedding
config["min_word_freq_threshold"] = args.min_word_freq_threshold
config["topk_word_freq_threshold"] = args.topk_word_freq_threshold
config["document_vector_agg_weight"] = args.document_vector_agg_weight # ['mean', 'IDF', 'uniform', 'gaussian', 'exponential', 'pmi']
config["select_topk_TFIDF"] = None
config["embedding_file"] = os.path.join("../data", args.embedding_file)
config["topk"] = args.topk

# In[3]:
sk_lasso_epoch = 10000
our_lasso_epoch = 50000
is_notebook = False


# load word embedding
embedding_file = config["embedding_file"]
word2embedding = dict()
word_dim = int(re.findall(r".(\d+)d",embedding_file)[0])

with open(embedding_file,"r") as f:
    for line in tqdm(f):
        line = line.strip().split()
        word = line[0]
        embedding = list(map(float,line[1:]))
        word2embedding[word] = np.array(embedding)

print("Number of words:%d" % len(word2embedding))


# In[4]:


def normalize_wordemb(word2embedding):
    word_emb = []
    word_list = []
    for word, emb in word2embedding.items():
        word_list.append(word)
        word_emb.append(emb)

    word_emb = np.array(word_emb)

    for i in range(len(word_emb)):
        norm = np.linalg.norm(word_emb[i])
        word_emb[i] = word_emb[i] / norm

    for word, emb in tqdm(zip(word_list, word_emb)):
        word2embedding[word] = emb
    return word2embedding

if config["normalize_word_embedding"]:
    normalize_wordemb(word2embedding)


# In[5]:


class Vocabulary:
    def __init__(self, word2embedding, min_word_freq_threshold=0, topk_word_freq_threshold=0):
        # The low frequency words will be assigned as <UNK> token
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}
        
        self.word2embedding = word2embedding
        self.min_word_freq_threshold = min_word_freq_threshold
        self.topk_word_freq_threshold = topk_word_freq_threshold
        
        self.word_freq_in_corpus = defaultdict(int)
        self.IDF = {}
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def __len__(self):
        return len(self.itos)

#     @staticmethod
    def tokenizer_eng(self, text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.strip().split()
        
        return [self.ps.stem(w) for w in text if w.lower() not in self.stop_words]

    def build_vocabulary(self, sentence_list):
        self.doc_freq = defaultdict(int) # # of document a word appear
        self.document_num = len(sentence_list)
        self.word_vectors = [[0]*word_dim] # unknown word emb
        
        for sentence in tqdm(sentence_list, desc="Preprocessing documents"):
            # for doc_freq
            document_words = set()
            
            for word in self.tokenizer_eng(sentence):
                # pass unknown word
                if word not in self.word2embedding:
                    continue
                    
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
            print(word)
            word = IDF[i][0]
            del self.IDF[word]
            del self.word_freq_in_corpus[word]
        
        # construct word_vectors
        idx = 1
        for word in self.word_freq_in_corpus:
            self.word_vectors.append(self.word2embedding[word])
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
            
    def init_word_weight(self,sentence_list, agg):
        if agg == 'mean':
            self.word_weight = {word: 1 for word in self.IDF.keys()}
        elif agg == 'IDF':
            self.word_weight = self.IDF
        elif agg == 'uniform':
            self.word_weight = {word: np.random.uniform(low=0.0, high=1.0) for word in self.IDF.keys()}
        elif agg == 'gaussian':
            mu, sigma = 10, 1 # mean and standard deviation
            self.word_weight = {word: np.random.normal(mu, sigma) for word in self.IDF.keys()}
        elif agg == 'exponential':
            self.word_weight = {word: np.random.exponential(scale=1.0) for word in self.IDF.keys()}
        elif agg == 'pmi':
            trigram_measures = BigramAssocMeasures()
            self.word_weight = defaultdict(int)
            corpus = []

            for text in tqdm(sentence_list):
                corpus.extend(text.split())

            finder = BigramCollocationFinder.from_words(corpus)
            for pmi_score in finder.score_ngrams(trigram_measures.pmi):
                pair, score = pmi_score
                self.word_weight[pair[0]] += score
                self.word_weight[pair[1]] += score
                
    def calculate_document_vector(self, sentence_list, agg, n_document, select_topk_TFIDF=None):
        document_vectors = []
        document_answers = []
        document_answers_w = []
        
        self.init_word_weight(sentence_list, agg)
        
        for sentence in tqdm(sentence_list[:min(n_document, len(sentence_list))], desc="calculate document vectors"):
            document_vector = np.zeros(len(self.word_vectors[0]))
            select_words = []
            for word in self.tokenizer_eng(sentence):
                # pass unknown word
                if word not in self.stoi:
                    continue
                else:
                    select_words.append(word)

            # select topk TDIDF
            if select_topk_TFIDF is not None:
                doc_TFIDF = defaultdict(float)
                for word in select_words:    
                    doc_TFIDF[word] += self.IDF[word]

                doc_TFIDF_l = [(word, TFIDF) for word, TFIDF in doc_TFIDF.items()]
                doc_TFIDF_l.sort(key=lambda x:x[1], reverse=True)
                
                select_topk_words = set(list(map(lambda x:x[0], doc_TFIDF_l[:select_topk_TFIDF])))
                select_words = [word for word in select_words if word in select_topk_words]
            else:
                pass
            
            total_weight = 0
            # aggregate to doc vectors
            for word in select_words:
                document_vector += np.array(self.word2embedding[word]) * self.word_weight[word]
                total_weight += self.word_weight[word]
                
            if len(select_words) == 0:
                print('error', sentence)
                continue
            else:
                document_vector /= total_weight
            
            document_vectors.append(document_vector)
            document_answers.append(select_words)
            document_answers_w.append(total_weight)
        
        # get answers
        document_answers_idx = []    
        for ans in document_answers:
            ans_idx = []
            for token in ans:
                if token in self.stoi:
                    ans_idx.append(self.stoi[token])                    
            document_answers_idx.append(ans_idx)

        return document_vectors, document_answers_idx, document_answers_w
        
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


# In[6]:


class CBowDataset(Dataset):
    def __init__(self, 
                 raw_data_file_path,
                 word2embedding,
                 skip_header = False,
                 n_document = None, # read first n document
                 min_word_freq_threshold = 20, # eliminate less freq words
                 topk_word_freq_threshold = 5, # eliminate smallest k IDF words
                 select_topk_TFIDF = None, # select topk tf-idf as ground-truth
                 document_vector_agg_weight = 'mean',
                 ):

        assert document_vector_agg_weight in ['mean', 'IDF', 'uniform', 'gaussian', 'exponential', 'pmi']
        
        # raw documents
        self.documents = []
        
        with open(raw_data_file_path,'r',encoding='utf-8') as f:
            if skip_header:
                f.readline()
            for line in tqdm(f, desc="Loading documents"):
                # read firt n document
                # if n_document is not None and len(self.documents) >= n_document:
                #     break    
                self.documents.append(line.strip("\n"))

        # build vocabulary
        self.vocab = Vocabulary(word2embedding, min_word_freq_threshold, topk_word_freq_threshold)
        self.vocab.build_vocabulary(self.documents)
        self.vocab_size = len(self.vocab)

        # calculate document vectors
        self.document_vectors, self.document_answers, self.document_answers_w = self.vocab.calculate_document_vector(self.documents,                                                                                            document_vector_agg_weight, n_document, select_topk_TFIDF)
                
        # train-test split
        # training
        self.train_split_ratio = 0.8
        self.train_length = int(len(self.document_answers) * self.train_split_ratio)
        self.train_vectors = self.document_vectors[:self.train_length]
        self.train_words = self.document_answers[:self.train_length]
        self.document_ids = list(range(self.train_length))
        self.generator = cycle(self.context_target_generator())
        self.dataset_size = sum([len(s) for s in self.train_words])
        
        # testing
        self.test_vectors = self.document_vectors[self.train_length:]
        self.test_words = self.document_answers[self.train_length:]

    def context_target_generator(self):
        np.random.shuffle(self.document_ids) # inplace shuffle

        # randomly select a document and create its training example
        for document_id in self.document_ids: 
            word_list = set(self.train_words[document_id])
            negative_sample_space = list(set(range(self.vocab_size)) - word_list)
            negative_samples = np.random.choice(negative_sample_space,size=len(word_list),replace = False)
            for word_id, negative_wordID in zip(word_list, negative_samples):
                yield [document_id, word_id, negative_wordID]
                
    def __getitem__(self, idx):
        doc_id, word_id, negative_wordID = next(self.generator)
        doc_id = torch.FloatTensor(self.document_vectors[doc_id])
        word_id = torch.FloatTensor(self.vocab.word_vectors[word_id])
        negative_word = torch.FloatTensor(self.vocab.word_vectors[negative_wordID])

        return doc_id, word_id, negative_word

    def __len__(self):
        return self.dataset_size 


# In[7]:


# load and build torch dataset
if config["dataset"] == 'IMDB':
    data_file_path = '../data/IMDB.txt'
elif config["dataset"] == 'CNN':
    data_file_path = '../data/CNN.txt'
elif config["dataset"] == 'PubMed':
    data_file_path = '../data/PubMed.txt'

print("Building dataset....")
dataset = CBowDataset(
                    raw_data_file_path=data_file_path,
                    word2embedding=word2embedding,
                    skip_header=False,
                    n_document = config["n_document"],
                    min_word_freq_threshold = config["min_word_freq_threshold"],
                    topk_word_freq_threshold = config["topk_word_freq_threshold"],
                    document_vector_agg_weight = config["document_vector_agg_weight"],
                    select_topk_TFIDF = config["select_topk_TFIDF"]
                    )


# In[8]:


print("Finish building dataset!")
print(f"Number of documents:{len(dataset.documents)}")
print(f"Number of words:{dataset.vocab_size}")

l = list(map(len, dataset.document_answers))
print("Average length of document:", np.mean(l))


# In[9]:


# check test doc vectors' correctness
word_vectors = np.array(dataset.vocab.word_vectors)
word_vectors.shape

pred = np.zeros(word_vectors.shape[1])
cnt = 0
for word_idx in dataset.test_words[0]:
    pred += word_vectors[word_idx] * dataset.vocab.word_weight[dataset.vocab.itos[word_idx]]
    cnt += dataset.vocab.word_weight[dataset.vocab.itos[word_idx]]
print(dataset.test_vectors[0] - pred/cnt)


# In[10]:


## create weight_ans
document_answers = dataset.document_answers

onehot_ans = np.zeros((len(document_answers), word_vectors.shape[0]))
weight_ans = np.zeros((len(document_answers), word_vectors.shape[0]))
print(weight_ans.shape)

for i in tqdm(range(len(document_answers))):
    for word_idx in document_answers[i]:
        weight_ans[i, word_idx] += dataset.vocab.word_weight[dataset.vocab.itos[word_idx]]
        onehot_ans[i, word_idx] += 1


# In[11]:


document_vectors = np.array(dataset.document_vectors)
document_answers_w = np.array(dataset.document_answers_w).reshape(-1, 1)


# ## Results

# In[12]:


final_results = []
select_columns = ['model']
for topk in config["topk"]:
    select_columns.append('F1@{}'.format(topk))
for topk in config["topk"]:
    select_columns.append('ndcg@{}'.format(topk))
select_columns.append('ndcg@all')
select_columns


# ## setting training size

# In[13]:


train_size_ratio = 1
train_size = int(len(dataset.document_answers) * train_size_ratio)
train_size


# ## Top K freq word

# In[14]:


topk_results = {}


# In[15]:


test_ans = dataset.document_answers[:train_size]


# In[16]:


word_freq = [(word, freq) for word, freq in dataset.vocab.word_freq_in_corpus.items()]
word_freq.sort(key=lambda x:x[1], reverse=True)
word_freq[:10]


# In[17]:


def topk_word_evaluation(k=50):
    topk_word = [word for (word, freq) in word_freq[:k]]

    pr, re = [], []
    for ans in tqdm(test_ans):
        ans = set(ans)
        ans = [dataset.vocab.itos[a] for a in ans]

        hit = []
        for word in ans:
            if word in topk_word:
                hit.append(word)

        precision = len(hit) / k
        recall = len(hit) / len(ans)
        pr.append(precision)
        re.append(recall)

    pr = np.mean(pr)
    re = np.mean(re)
    f1 = 2 * pr * re / (pr + re) if (pr + re) != 0 else 0
    print('top {} word'.format(k))
    print('percision', np.mean(pr))
    print('recall', np.mean(re))
    print('F1', f1)
    return f1


for topk in config['topk']:
    topk_results["F1@{}".format(topk)] = topk_word_evaluation(k=topk)


# In[18]:


def topk_word_evaluation_NDCG(k=50):
    freq_word =[word for (word, freq) in word_freq]
    freq_word_idx = [dataset.vocab.stoi[word] for word in freq_word if word in dataset.vocab.stoi]
    
    scores = np.zeros(len(dataset.vocab.word_vectors))
    for rank, idx in enumerate(freq_word_idx):
        scores[idx] = len(dataset.vocab.word_vectors) - rank
    
    NDCGs = []
    
    for ans in tqdm(test_ans):
        weight_ans = np.zeros(len(dataset.vocab.word_vectors))
        
        for word_idx in ans:
            if word_idx == 0:
                continue
            word = dataset.vocab.itos[word_idx]
            weight_ans[word_idx] += dataset.vocab.IDF[word]

        NDCG_score = ndcg_score(weight_ans.reshape(1,-1), scores.reshape(1,-1), k=k)
        NDCGs.append(NDCG_score)

    print('top {} NDCG:{}'.format(k, np.mean(NDCGs)))
    
    return np.mean(NDCGs)


for topk in config['topk']:
    topk_results["ndcg@{}".format(topk)] = topk_word_evaluation_NDCG(k=topk)
    
topk_results["ndcg@all"] = topk_word_evaluation_NDCG(k=None)


# In[19]:


topk_results["model"] = "topk"
final_results.append(pd.Series(topk_results))


# ## Sklearn

# In[20]:


from sklearn.linear_model import LinearRegression, Lasso


# In[21]:


print(document_vectors.shape)
print(weight_ans.shape)
print(word_vectors.shape)


# In[22]:


def evaluate_sklearn(pred, ans):
    results = {}
        
    one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    
    for topk in config["topk"]:
        one_hot_pred = np.argsort(pred)[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        recall = len(hit) / len(one_hot_ans)
        f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
        
        results['F1@{}'.format(topk)] = f1
        
    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
    return results


# ### linear regression

# In[23]:


results = []

for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
    x = word_vectors.T
    y = doc_emb
    
    ans = weight_ans[doc_id]
    model = LinearRegression(fit_intercept=False).fit(x, y)
    r2 = model.score(x, y)

    res = evaluate_sklearn(model.coef_, ans)
    results.append(res)


# In[24]:


results = pd.DataFrame(results).mean()
results['model'] = 'sk-linear-regression'
final_results.append(results)
results


# In[25]:


results = []

for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
    x = word_vectors.T
    y = doc_emb
    
    ans = weight_ans[doc_id]
    model = Lasso(positive=True, fit_intercept=False, alpha=0.0001, max_iter=sk_lasso_epoch, tol=0).fit(x, y)
    r2 = model.score(x, y)

    res = evaluate_sklearn(model.coef_, ans)
    results.append(res)


# In[26]:


results = pd.DataFrame(results).mean()
results['model'] = 'sk-lasso'
final_results.append(results)
results


# ## Our Model

# In[27]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[28]:


class Custom_Dataset(Dataset):
    def __init__(self, 
                 doc_vectors,
                 doc_w_sum,
                 weight_ans
                 ):
        self.doc_vectors = torch.FloatTensor(doc_vectors)
        self.doc_w_sum = torch.FloatTensor(doc_w_sum)
        self.weight_ans = weight_ans
        assert len(doc_vectors) == len(doc_w_sum)
        
    def __getitem__(self, idx):
                
        return self.doc_vectors[idx], self.doc_w_sum[idx], idx

    def __len__(self):
        return len(self.doc_vectors)


# In[29]:


class LR(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, num_doc, num_words):
        super(LR, self).__init__()
        weight = torch.zeros(num_doc, num_words).to(device)
        self.emb = torch.nn.Embedding.from_pretrained(weight, freeze=False)
        
    def forward(self, doc_ids, word_vectors):
        return self.emb(doc_ids) @ word_vectors


# In[30]:


def evaluate_NDCG(model, train_loader):
    results = {}
    model.eval()
    
    scores = np.array(model.emb.cpu().weight.data)
    model.emb.to(device)
    true_relevance = train_loader.dataset.weight_ans

    # F1
    F1s = []
    for i in range(true_relevance.shape[0]):
        one_hot_ans = np.arange(true_relevance.shape[1])[true_relevance[i] > 0]
        pred = scores[i]
        
        F1 = []
        for topk in config["topk"]:
            one_hot_pred = np.argsort(pred)[-topk:]
            
            hit = np.intersect1d(one_hot_pred, one_hot_ans)
            percision = len(hit) / topk
            recall = len(hit) / len(one_hot_ans)
            
            ans = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
            F1.append(ans)
        F1s.append(F1)
        
    F1s = np.mean(F1s, axis=0)
    
    for i, topk in enumerate(config["topk"]):
        results['F1@{}'.format(topk)] = F1s[i]

    # NDCG
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(true_relevance, scores, k=topk)
    results['ndcg@all'] = ndcg_score(true_relevance, scores, k=None)
    
    return results


# In[31]:


batch_size = 100
print('document num', train_size)

train_dataset = Custom_Dataset(document_vectors[:train_size], document_answers_w[:train_size], weight_ans[:train_size])
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


# ## start training

# In[32]:


# setting
lr = 0.5
momentum = 0.99
weight_decay = 0
nesterov = False # True

n_epoch = our_lasso_epoch

w_sum_reg = 1e-3
w_sum_reg_mul = 0.9
w_clip_value = 0

L1 = 1e-6

verbose = False
valid_epoch = 100

model = LR(num_doc=train_size, num_words=word_vectors.shape[0]).to(device)
model.train()

word_vectors_tensor = torch.FloatTensor(word_vectors).to(device)
    
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
criterion = nn.MSELoss(reduction='mean')

results = []
step = 0
for epoch in tqdm(range(n_epoch)):    
    loss_mse_his = []
    loss_w_reg_his = []
    
    model.train()

    for data in train_loader:
        doc_embs, doc_w_sum, doc_ids = data
        
        doc_embs = doc_embs.to(device)
        doc_w_sum = doc_w_sum.to(device)
        doc_ids = doc_ids.to(device)
        
        w_reg = (torch.ones(doc_embs.size(0), 1) * w_sum_reg_mul).to(device)
        
        # MSE loss
        pred_doc_embs = model(doc_ids, word_vectors_tensor)     
        loss_mse = criterion(pred_doc_embs, doc_embs)

        pred_w_sum = torch.sum(model.emb(doc_ids), axis=1).view(-1, 1)
        loss_w_reg = criterion(pred_w_sum, w_reg)
        
        loss_l1 = torch.sum(torch.abs(model.emb(doc_ids)))
        loss = loss_mse + loss_w_reg * w_sum_reg + loss_l1 * L1
        
        # Model backwarding
        model.zero_grad()
        loss.backward()
        opt.step()

        loss_mse_his.append(loss_mse.item())
        loss_w_reg_his.append(loss_w_reg.item())

        for p in model.parameters():
            p.data.clamp_(w_clip_value, float('inf'))

        
    if epoch % valid_epoch == 0:
        res = {}
        res['epoch'] = epoch
        res['loss_mse'] = np.mean(loss_mse_his)
        res['loss_w_reg'] = np.mean(loss_w_reg_his)
        
        res_ndcg = evaluate_NDCG(model, train_loader)
        res.update(res_ndcg)
        results.append(res)
        
        if verbose:
            print()
            for k, v in res.items():
                print(k, v)


# In[33]:


pd.set_option('display.max_rows', 500)
results_df = pd.DataFrame(results).set_index('epoch')
results_df


# In[34]:


results_df['model'] = 'our-lasso'
final_results.append(results_df[select_columns].iloc[-1])


# ## Quality Check

# In[108]:


# select doc_id and k
doc_id = 90
topk = 30

model


# In[109]:


import colored
from colored import stylize

word_list = dataset.vocab.itos

gt = [word_list[word_idx] for word_idx in np.argsort(weight_ans[doc_id])[::-1][:topk]]
pred = [word_list[word_idx] for word_idx in np.argsort(model.emb.cpu().weight.data[doc_id].numpy())[::-1][:topk]]

print('ground truth')
for word in gt:
    if word in pred:
        print(stylize(word, colored.bg("yellow")), end=' ')
    else:
        print(word, end=' ')

print()
print('\nprediction')
for word in pred:
    if word in gt:
        print(stylize(word, colored.bg("yellow")), end=' ')
    else:
        print(word, end=' ')


# In[110]:


# raw document
print()
ps = PorterStemmer()
    
for word in dataset.documents[doc_id].split():
    word_stem = ps.stem(word)
    if word_stem in gt:
        if word_stem in pred:
            print(stylize(word, colored.bg("yellow")), end=' ')
        else:
            print(stylize(word, colored.bg("light_gray")), end=' ')
    else:
        print(word, end=' ')
# print(dataset.documents[doc_id])


# In[111]:


results = {}
   
scores = np.array(model.emb.weight.data)[doc_id].reshape(1, -1)
true_relevance = train_loader.dataset.weight_ans[doc_id].reshape(1, -1)

results['ndcg@50'] = (ndcg_score(true_relevance, scores, k=50))
results['ndcg@100'] = (ndcg_score(true_relevance, scores, k=100))
results['ndcg@200'] = (ndcg_score(true_relevance, scores, k=200))
results['ndcg@all'] = (ndcg_score(true_relevance, scores, k=None))

print('This document ndcg:')
print('ground truth length:', np.sum(weight_ans[doc_id] > 0))
print('NDCG top50', results['ndcg@50'])
print('NDCG top100', results['ndcg@100'])
print('NDCG top200', results['ndcg@200'])
print('NDCG ALL', results['ndcg@all'])


# ## Final results

# In[39]:


final_results_df = pd.DataFrame(final_results).reset_index(drop=True)

experiment_dir = './records/dataset-{}-n_document-{}-wdist-{}-filtertopk-{}'.format(
                                        config['dataset'],
                                        config['n_document'],
                                        config["document_vector_agg_weight"],
                                        config["topk_word_freq_threshold"])

print('Saving to directory', experiment_dir)
os.makedirs(experiment_dir, exist_ok=True)


# In[40]:


final_results_df.to_csv(os.path.join(experiment_dir, 'result.csv'), index=False)

import json
with open(os.path.join(experiment_dir, 'config.json'), 'w') as json_file:
    json.dump(config, json_file)


# In[114]:


for feat in final_results_df.set_index('model').columns:
    plt.bar(final_results_df['model'],
            final_results_df[feat], 
            width=0.5, 
            bottom=None, 
            align='center', 
            color=['lightsteelblue', 
                   'cornflowerblue', 
                   'royalblue', 
                   'navy'])
    plt.title(feat)
    plt.savefig(os.path.join(experiment_dir, '{}.png'.format(feat)))
    plt.clf()
    if is_notebook:
        plt.show()


# In[42]:


print(final_results_df)


# In[ ]:




