import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
from collections import defaultdict
import math
import numpy as np 
import random
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
import warnings
warnings.filterwarnings("ignore")

seed = 33
import pandas as pd
import time

# ## Preprocess config

import argparse
parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="CNN")
parser.add_argument('--no_sklearn', action='store_true')
parser.add_argument('--no_bp', action='store_true')
parser.add_argument('--n_document', type=int, default=1e9)
parser.add_argument('--min_word_freq_threshold', type=int, default=20)
parser.add_argument('--topk_word_freq_threshold', type=int, default=100)
parser.add_argument('--document_vector_agg_weight', type=str, default='IDF')
parser.add_argument('--normalize_word_embedding', action='store_true')
parser.add_argument('--no_document_vector_weight_normalize', action='store_false')
parser.add_argument('--embedding_file', type=str, default='glove.6B.100d.txt')
parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])

parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.999)
parser.add_argument('--L1', type=float, default=1e-5)
parser.add_argument('--w_sum_reg', type=float, default=1e-2)
parser.add_argument('--w_sum_reg_mul', type=float, default=1)
parser.add_argument('--w_clip_value', type=float, default=0)


parser.add_argument('--lasso_epochs', type=int, default=1000)
parser.add_argument('--bpdn_epochs', type=int, default=200)

args = parser.parse_args()
# In[2]:


config = {}

config["dataset"] = args.dataset # "IMDB" "CNN", "PubMed"
config["n_document"] = args.n_document
config["normalize_word_embedding"] = args.normalize_word_embedding
print(config["normalize_word_embedding"])
config["min_word_freq_threshold"] = args.min_word_freq_threshold
config["topk_word_freq_threshold"] = args.topk_word_freq_threshold
config["document_vector_agg_weight"] = args.document_vector_agg_weight # ['mean', 'IDF', 'uniform', 'gaussian', 'exponential', 'pmi']
config["document_vector_weight_normalize"] = args.no_document_vector_weight_normalize # weighted sum or mean, True for mean, False for sum 
config["select_topk_TFIDF"] = None
config["embedding_file"] = os.path.join("../data", args.embedding_file)
config["topk"] = args.topk

# In[2]:


# config = {}

# config["dataset"] = "CNN" # "IMDB" "CNN", "PubMed"
# config["n_document"] = 100
# config["normalize_word_embedding"] = False
# config["min_word_freq_threshold"] = 20
# config["topk_word_freq_threshold"] = 100
# config["document_vector_agg_weight"] = 'IDF' # ['mean', 'IDF', 'uniform', 'gaussian', 'exponential', 'pmi']
# config["document_vector_weight_normalize"] = True # weighted sum or mean, True for mean, False for sum 
# config["select_topk_TFIDF"] = None # ignore
# config["embedding_file"] = "../data/glove.6B.100d.txt"
# config["topk"] = [10, 30, 50]


# In[3]:


def in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
    except ImportError:
        return False
    return True


# In[4]:


def load_word2emb(embedding_file):
    word2embedding = dict()
    word_dim = int(re.findall(r".(\d+)d", embedding_file)[0])

    with open(embedding_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            word2embedding[word] = np.array(embedding)

    print("Number of words:%d" % len(word2embedding))

    return word2embedding

word2embedding = load_word2emb(config["embedding_file"])


# In[5]:


def normalize_wordemb(word2embedding):
    # Every word emb should have norm 1
    
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


# In[6]:


class Vocabulary:
    def __init__(self, word2embedding, config):
        # The low frequency words will be assigned as <UNK> token
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}
        
        self.word2embedding = word2embedding
        self.config = config

        self.word_freq_in_corpus = defaultdict(int)
        self.IDF = {}
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        self.word_dim = len(word2embedding['the'])
    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.strip().split()
        
        return [self.ps.stem(w) for w in text if w.lower() not in self.stop_words]
    
    def read_raw(self):        
        if self.config["dataset"] == 'IMDB':
            data_file_path = '../data/IMDB.txt'
        elif self.config["dataset"] == 'CNN':
            data_file_path = '../data/CNN.txt'
        elif self.config["dataset"] == 'PubMed':
            data_file_path = '../data/PubMed.txt'
        
        # raw documents
        self.raw_documents = []
        with open(data_file_path,'r',encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                self.raw_documents.append(line.strip("\n"))
                
        return self.raw_documents
    
    def build_vocabulary(self):
        sentence_list = self.raw_documents
        
        self.doc_freq = defaultdict(int) # # of document a word appear
        self.document_num = len(sentence_list)
        self.word_vectors = [[0]*self.word_dim] # unknown word emb
        
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
            if v < self.config["min_word_freq_threshold"]:
                delete_words.append(word)     
        for word in delete_words:
            del self.IDF[word]    
            del self.word_freq_in_corpus[word]    
        
        # delete too freq words
        print('eliminate freq words')
        IDF = [(word, freq) for word, freq in self.IDF.items()]
        IDF.sort(key=lambda x: x[1])

        for i in range(self.config["topk_word_freq_threshold"]):
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
                
    def calculate_document_vector(self):
        # Return
        # document_vectors: weighted sum of word emb
        # document_answers_idx: doc to word index list
        # document_answers_wsum: word weight summation, e.g. total TFIDF score of a doc
        
        document_vectors = [] 
        document_answers = []
        document_answers_wsum = []
        
        sentence_list = self.raw_documents
        agg = self.config["document_vector_agg_weight"]
        n_document = self.config["n_document"]
        select_topk_TFIDF = self.config["select_topk_TFIDF"]
        
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
                
            if len(select_words) <= 5:
                print('error', sentence)
                continue
            else:
                if self.config["document_vector_weight_normalize"]:
                    document_vector /= total_weight
                    total_weight = 1
            
            document_vectors.append(document_vector)
            document_answers.append(select_words)
            document_answers_wsum.append(total_weight)
        
        # get answers
        document_answers_idx = []    
        for ans in document_answers:
            ans_idx = []
            for token in ans:
                if token in self.stoi:
                    ans_idx.append(self.stoi[token])                    
            document_answers_idx.append(ans_idx)
        
        self.document_vectors = document_vectors
        self.document_answers_idx = document_answers_idx
        self.document_answers_wsum = document_answers_wsum
        
        return document_vectors, document_answers_idx, document_answers_wsum
        
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    
    def check_docemb(self):
        word_vectors = np.array(self.word_vectors)
        pred = np.zeros(word_vectors.shape[1])
        cnt = 0

        for word_idx in self.document_answers_idx[0]:
            pred += word_vectors[word_idx] * self.word_weight[self.itos[word_idx]]
            cnt += self.word_weight[self.itos[word_idx]]
        
        if self.config["document_vector_weight_normalize"]:
            pred /= cnt
        assert np.sum(self.document_vectors[0]) - np.sum(pred) == 0


# In[7]:


def build_vocab(config, word2embedding):
    # build vocabulary
    vocab = Vocabulary(word2embedding, config)
    vocab.read_raw()
    vocab.build_vocabulary()
    vocab_size = len(vocab)
    # get doc emb
    vocab.calculate_document_vector()
    vocab.check_docemb()
    
    return vocab

vocab = build_vocab(config, word2embedding)


# In[8]:


print("Finish building dataset!")
print(f"Number of documents:{len(vocab.raw_documents)}")
print(f"Number of words:{len(vocab)}")

l = list(map(len, vocab.document_answers_idx))
print("Average length of document:", np.mean(l))


# In[9]:


word_vectors = np.array(vocab.word_vectors)
print("word_vectors:", word_vectors.shape)

document_vectors = np.array(vocab.document_vectors)
print("document_vectors", document_vectors.shape)

document_answers_wsum = np.array(vocab.document_answers_wsum).reshape(-1, 1)
print("document_answers_wsum", document_answers_wsum.shape)

# create weight_ans
document_answers_idx = vocab.document_answers_idx

# random shuffle
shuffle_idx = list(range(len(document_vectors)))
random.Random(seed).shuffle(shuffle_idx)

document_vectors = document_vectors[shuffle_idx]
document_answers_wsum = document_answers_wsum[shuffle_idx]
document_answers_idx = [document_answers_idx[idx] for idx in shuffle_idx]


# In[10]:


# onthot_ans: word freq matrix
# weight_ans: TFIDF matrix

onehot_ans = np.zeros((len(document_answers_idx), word_vectors.shape[0]))
weight_ans = np.zeros((len(document_answers_idx), word_vectors.shape[0]))
print(weight_ans.shape)

for i in tqdm(range(len(document_answers_idx))):
    for word_idx in document_answers_idx[i]:
        weight_ans[i, word_idx] += vocab.word_weight[vocab.itos[word_idx]]
        onehot_ans[i, word_idx] += 1
        
    if config["document_vector_weight_normalize"]:
        weight_ans[i] /= np.sum(weight_ans[i])


# In[11]:


# check
assert np.sum(document_vectors - np.dot(weight_ans, word_vectors) > 1e-10) == 0


# ## Results

# In[12]:


final_results = []
select_columns = ['model']
for topk in config["topk"]:
    select_columns.append('percision@{}'.format(topk))
for topk in config["topk"]:
    select_columns.append('recall@{}'.format(topk))
for topk in config["topk"]:
    select_columns.append('F1@{}'.format(topk))
for topk in config["topk"]:
    select_columns.append('ndcg@{}'.format(topk))
select_columns.append('ndcg@all')
select_columns


# ## setting training size

# In[13]:


train_size_ratio = 1
train_size = int(len(document_answers_idx) * train_size_ratio)
train_size


# ## Top K freq word

# In[14]:


topk_results = {}


# In[15]:


test_ans = document_answers_idx[:train_size]


# In[16]:


word_freq = [(word, freq) for word, freq in vocab.word_freq_in_corpus.items()]
word_freq.sort(key=lambda x:x[1], reverse=True)
word_freq[:10]


# In[17]:


def topk_word_evaluation(k=50):
    topk_word = [word for (word, freq) in word_freq[:k]]

    pr, re = [], []
    for ans in tqdm(test_ans):
        ans = set(ans)
        ans = [vocab.itos[a] for a in ans]

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
    freq_word_idx = [vocab.stoi[word] for word in freq_word if word in vocab.stoi]
    
    scores = np.zeros(len(vocab.word_vectors))
    for rank, idx in enumerate(freq_word_idx):
        scores[idx] = len(vocab.word_vectors) - rank
    
    NDCGs = []
    
    for ans in tqdm(test_ans):
        weight_ans = np.zeros(len(vocab.word_vectors))
        
        for word_idx in ans:
            if word_idx == 0:
                continue
            word = vocab.itos[word_idx]
            weight_ans[word_idx] += vocab.IDF[word]

        NDCG_score = ndcg_score(weight_ans.reshape(1,-1), scores.reshape(1,-1), k=k)
        NDCGs.append(NDCG_score)

    print('top {} NDCG:{}'.format(k, np.mean(NDCGs)))
    
    return np.mean(NDCGs)


# for topk in config['topk']:
#     topk_results["ndcg@{}".format(topk)] = topk_word_evaluation_NDCG(k=topk)
    
# topk_results["ndcg@all"] = topk_word_evaluation_NDCG(k=None)


# In[19]:


topk_results["model"] = "topk"
# final_results.append(pd.Series(topk_results))


# ## Sklearn

# In[20]:


from sklearn.linear_model import LinearRegression, Lasso


# In[21]:


print(document_vectors.shape)
print(weight_ans.shape)
print(word_vectors.shape)


# In[22]:


# def evaluate_sklearn(pred, ans):
#     results = {}
        
#     one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    
#     for topk in config["topk"]:
#         one_hot_pred = np.argsort(pred)[-topk:]
#         hit = np.intersect1d(one_hot_pred, one_hot_ans)
#         percision = len(hit) / topk
#         recall = len(hit) / len(one_hot_ans)
#         f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
        
#         results['percision@{}'.format(topk)] = percision
#         results['recall@{}'.format(topk)] = recall
#         results['F1@{}'.format(topk)] = f1
        
#     ans = ans.reshape(1, -1)
#     pred = pred.reshape(1, -1)
#     for topk in config["topk"]:
#         results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

#     results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
#     return results


def evaluate_sklearn(pred, ans):
    results = {}
        
    # one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    
    for topk in config["topk"]:
        topk_ = topk
        topk = min(topk, sum(ans > 0))
        one_hot_pred = np.argsort(pred)[-topk:]
        one_hot_ans = np.argsort(ans)[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        recall = len(hit) / len(one_hot_ans)
        f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
        
        results['percision@{}'.format(topk_)] = percision
        results['recall@{}'.format(topk_)] = recall
        results['F1@{}'.format(topk_)] = f1
        
    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
    return results

# ### linear regression

start_time = time.time()

# In[23]:
if not args.no_sklearn:

    results = []

    for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
        x = word_vectors.T
        y = doc_emb
        
        ans = weight_ans[doc_id]
        model = LinearRegression(fit_intercept=False).fit(x, y)
        r2 = model.score(x, y)

        # res = evaluate_sklearn(model.coef_, ans)
        # results.append(res)


    total_seconds = time.time() - start_time
    # In[24]:


    results = pd.DataFrame(results).mean()
    results['model'] = 'sk-linear-regression'
    results['epochs'] = None
    results['total_seconds'] = total_seconds
    final_results.append(results)
    results


# ### lasso

# In[25]:

start_time = time.time()

if not args.no_sklearn:
    
    results = []
    sk_lasso_epoch = args.lasso_epochs

    for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
        x = word_vectors.T
        y = doc_emb
        
        ans = weight_ans[doc_id]
        model = Lasso(positive=True, fit_intercept=False, alpha=0.0001, max_iter=sk_lasso_epoch, tol=0).fit(x, y)
        r2 = model.score(x, y)

        # res = evaluate_sklearn(model.coef_, ans)
        # results.append(res)


    total_seconds = time.time() - start_time
    # In[26]:


    results = pd.DataFrame(results).mean()
    results['model'] = 'sk-lasso'
    results['epochs'] = sk_lasso_epoch
    results['total_seconds'] = total_seconds
    final_results.append(results)
    results


from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV


start_time = time.time()

if not args.no_bp:
    results = []

    for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
        x = word_vectors.T
        y = doc_emb
        
        ans = weight_ans[doc_id]
        n_nonzero_coefs = np.sum(weight_ans[doc_id] > 0) // 1
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=None, normalize=True, fit_intercept=False)
        omp.fit(x, y)

        # res = evaluate_sklearn(omp.coef_, ans)
        # results.append(res)

    total_seconds = time.time() - start_time

    results = pd.DataFrame(results).mean()
    results['model'] = 'sk-omp'
    results['epochs'] = None
    results['total_seconds'] = total_seconds
    final_results.append(results)
    results


import pylops

start_time = time.time()

if not args.no_bp:
    results = []

    for doc_id, doc_emb in enumerate(tqdm(document_vectors[:train_size])):
        d = doc_emb
        
        ans = weight_ans[doc_id]
        xspgl1, pspgl1, info = \
            pylops.optimization.sparsity.SPGL1(word_vectors.T, d, None, tau=0, sigma=1e-3, iter_lim=args.bpdn_epochs)

        # res = evaluate_sklearn(xspgl1, ans)
        # results.append(res)

    total_seconds = time.time() - start_time

    results = pd.DataFrame(results).mean()
    results['model'] = 'BPDN'
    results['epochs'] = args.bpdn_epochs
    results['total_seconds'] = total_seconds
    final_results.append(results)
    results
# ## Our Model

# In[27]:


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


# In[28]:


class Custom_Lasso_Dataset(Dataset):
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


# def evaluate_Custom_Lasso(model, train_loader):
#     results = {}
#     model.eval()
    
#     scores = np.array(model.emb.cpu().weight.data)
#     model.emb.to(device)
#     true_relevance = train_loader.dataset.weight_ans

#     # F1
#     F1s = []
#     precisions = []
#     recalls = []
#     for i in range(true_relevance.shape[0]):
#         one_hot_ans = np.arange(true_relevance.shape[1])[true_relevance[i] > 0]
#         pred = scores[i]
        
#         F1_ = []
#         percision_ = []
#         recall_ = []
#         for topk in config["topk"]:
#             one_hot_pred = np.argsort(pred)[-topk:]
            
#             hit = np.intersect1d(one_hot_pred, one_hot_ans)
#             percision = len(hit) / topk
#             recall = len(hit) / len(one_hot_ans)
            
#             F1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
#             F1_.append(F1)
#             percision_.append(percision)
#             recall_.append(recall)
            
#         F1s.append(F1_)
#         precisions.append(percision_)
#         recalls.append(recall_)
        
#     F1s = np.mean(F1s, axis=0)
#     precisions = np.mean(precisions, axis=0)
#     recalls = np.mean(recalls, axis=0)
    
#     for i, topk in enumerate(config["topk"]):
#         results['F1@{}'.format(topk)] = F1s[i]
#         results['percision@{}'.format(topk)] = precisions[i]
#         results['recall@{}'.format(topk)] = recalls[i]

#     # NDCG
#     for topk in config["topk"]:
#         results['ndcg@{}'.format(topk)] = ndcg_score(true_relevance, scores, k=topk)
#     results['ndcg@all'] = ndcg_score(true_relevance, scores, k=None)
    
#     return results

def evaluate_Custom_Lasso(model, train_loader):
    results = {}
    model.eval()
    
    scores = np.array(model.emb.cpu().weight.data)
    model.emb.to(device)
    true_relevance = train_loader.dataset.weight_ans

    # F1
    F1s = []
    precisions = []
    recalls = []
    for i in range(true_relevance.shape[0]):
        one_hot_ans = np.arange(true_relevance.shape[1])[true_relevance[i] > 0]
        pred = scores[i]
        
        F1_ = []
        percision_ = []
        recall_ = []
        for topk in config["topk"]:
            topk = min(topk, sum(true_relevance[i] > 0))
            one_hot_pred = np.argsort(pred)[-topk:]
            one_hot_ans = np.argsort(true_relevance[i])[-topk:]
            
            hit = np.intersect1d(one_hot_pred, one_hot_ans)
            percision = len(hit) / topk
            recall = len(hit) / len(one_hot_ans)
            
            F1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
            F1_.append(F1)
            percision_.append(percision)
            recall_.append(recall)
            
        F1s.append(F1_)
        precisions.append(percision_)
        recalls.append(recall_)
        
    F1s = np.mean(F1s, axis=0)
    precisions = np.mean(precisions, axis=0)
    recalls = np.mean(recalls, axis=0)
    
    for i, topk in enumerate(config["topk"]):
        results['F1@{}'.format(topk)] = F1s[i]
        results['percision@{}'.format(topk)] = precisions[i]
        results['recall@{}'.format(topk)] = recalls[i]

    # NDCG
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(true_relevance, scores, k=topk)
    results['ndcg@all'] = ndcg_score(true_relevance, scores, k=None)
    
    return results
# In[31]:


batch_size = 100
print('document num', train_size)

train_dataset = Custom_Lasso_Dataset(document_vectors[:train_size], document_answers_wsum[:train_size], weight_ans[:train_size])
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


# ## start training

# In[32]:

# def train_CLasso(select_idx):
#     batch_size = len(select_idx)
#     print('batch_size', batch_size)
#     train_dataset = Custom_Lasso_Dataset(document_vectors[select_idx], document_answers_wsum[select_idx], weight_ans[select_idx])
#     train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
#     model = LR(num_doc=batch_size, num_words=word_vectors.shape[0]).to(device)
#     model.train()

#     word_vectors_tensor = torch.FloatTensor(word_vectors).to(device)

#     opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
#     criterion = nn.MSELoss(reduction='mean')

#     results = []
#     step = 0
#     for epoch in tqdm(range(n_epoch)):    
#         loss_mse_his = []
#         loss_w_reg_his = []

#         model.train()

#         for data in train_loader:
#             doc_embs, doc_w_sum, doc_ids = data

#             doc_embs = doc_embs.to(device)
#             doc_w_sum = doc_w_sum.to(device)
#             doc_ids = doc_ids.to(device)

#             w_reg = doc_w_sum * w_sum_reg_mul
#             # w_reg = (torch.ones(doc_embs.size(0), 1) * w_sum_reg_mul).to(device)

#             # MSE loss
#             pred_doc_embs = model(doc_ids, word_vectors_tensor)     
#             loss_mse = criterion(pred_doc_embs, doc_embs)

#             pred_w_sum = torch.sum(model.emb(doc_ids), axis=1).view(-1, 1)
#             loss_w_reg = criterion(pred_w_sum, w_reg)

#             loss_l1 = torch.sum(torch.abs(model.emb(doc_ids)))
#             loss = loss_mse + loss_w_reg * w_sum_reg + loss_l1 * L1

#             # Model backwarding
#             model.zero_grad()
#             loss.backward()
#             opt.step()

#             loss_mse_his.append(loss_mse.item())
#             loss_w_reg_his.append(loss_w_reg.item())

#             for p in model.parameters():
#                 p.data.clamp_(w_clip_value, float('inf'))


#         # if (verbose and epoch % valid_epoch == 0) or (not verbose and epoch == n_epoch-1):
#         #     res = {}
#         #     res['epoch'] = epoch
#         #     res['loss_mse'] = np.mean(loss_mse_his)
#         #     res['loss_w_reg'] = np.mean(loss_w_reg_his)

#         #     res_ndcg = evaluate_Custom_Lasso(model, train_loader)
#         #     res.update(res_ndcg)
#         #     results.append(res)

#         #     print()
#         #     for k, v in res.items():
#         #         print(k, v)
                
#     return pd.DataFrame([]), model

def train_CLasso(select_idx):
    batch_size = len(select_idx)
    print('batch_size', batch_size)
    train_dataset = Custom_Lasso_Dataset(document_vectors[select_idx], document_answers_wsum[select_idx], weight_ans[select_idx])
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = LR(num_doc=batch_size, num_words=word_vectors.shape[0]).to(device)
    model.train()

    word_vectors_tensor = torch.FloatTensor(word_vectors).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    criterion = nn.MSELoss(reduction='none')

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

            w_reg = doc_w_sum * w_sum_reg_mul
            # w_reg = (torch.ones(doc_embs.size(0), 1) * w_sum_reg_mul).to(device)

            # MSE loss
            pred_doc_embs = model(doc_ids, word_vectors_tensor)     
            loss_mse = criterion(pred_doc_embs, doc_embs)
            loss_mse = torch.sum(torch.mean(loss_mse, dim=1))

            pred_w_sum = torch.sum(model.emb(doc_ids), axis=1).view(-1, 1)
            loss_w_reg = criterion(pred_w_sum, w_reg)
            loss_w_reg = torch.sum(torch.mean(loss_w_reg, dim=1))

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
#                 p.data -= L1
#                 p.data[p.data < 0] = 0


        if (verbose and epoch % valid_epoch == 0) or (not verbose and epoch == n_epoch-1):
            res = {}
            res['epoch'] = epoch
            res['loss_mse'] = np.mean(loss_mse_his)
            res['loss_w_reg'] = np.mean(loss_w_reg_his)

            res_ndcg = evaluate_Custom_Lasso(model, train_loader)
            res.update(res_ndcg)
            results.append(res)

            print()
            for k, v in res.items():
                print(k, v)
                
    return res, model
    
# setting
lr = args.lr
momentum = args.momentum
weight_decay = 0
nesterov = False # True

n_epoch = args.epochs

w_sum_reg = args.w_sum_reg
w_sum_reg_mul = args.w_sum_reg_mul
w_clip_value = args.w_clip_value

L1 = args.L1

verbose = False
valid_epoch = 100

start_time = time.time()

results = []
for i in range(len(document_vectors)//batch_size+1):
    if i != len(document_vectors)//batch_size:
        select_idx = np.arange(i*batch_size, (i+1)*batch_size)
    elif len(document_vectors)%batch_size != 0:
        select_idx = np.arange(i*batch_size, len(document_vectors))
    else:
        break
    res, model = train_CLasso(select_idx)
    results.append(res)

total_seconds = time.time() - start_time
# In[33]:


pd.set_option('display.max_rows', 500)
results_df = pd.DataFrame([])
results_df

# In[34]:

results_df['model'] = 'our-lasso'
results_df['epochs'] = n_epoch
results_df['total_seconds'] = total_seconds
select_columns = ['model', 'epochs', 'total_seconds']

# ## Quality Check

# In[35]:



# ## Final results

# In[39]:


is_notebook = in_notebook()


# In[40]:
import time
t = time.localtime()
t = time.strftime("%Y-%m-%d_%H:%M:%S", t)

final_results_df = pd.DataFrame(final_results).reset_index(drop=True)

s = pd.Series({'model':'our-lasso', 'epochs':n_epoch, 'total_seconds':total_seconds})
final_results_df = final_results_df.append(s, ignore_index=True)
# final_results_df = final_results_df.loc[4] = ['our-lasso'. n_epoch, total_seconds]
print(final_results_df)

experiment_dir = './records_time2/dataset-{}-n_document-{}-wdist-{}-filtertopk-{}-dim-{}-epoch-{}-time-{}'.format(
                                        config['dataset'],
                                        config['n_document'],
                                        config["document_vector_agg_weight"],
                                        config["topk_word_freq_threshold"],
                                        config["embedding_file"].split('.')[-2][:-1],
                                        args.epochs,
                                        t)

print('Saving to directory', experiment_dir)
os.makedirs(experiment_dir, exist_ok=True)

# In[41]:
final_results_df.to_csv(os.path.join(experiment_dir, 'result.csv'), index=False)

import json
with open(os.path.join(experiment_dir, 'config.json'), 'w') as json_file:
    json.dump(vars(args), json_file)

# In[42]:


# for feat in final_results_df.set_index('model').columns:
#     plt.bar(final_results_df['model'],
#             final_results_df[feat], 
#             width=0.5, 
#             bottom=None, 
#             align='center', 
#             color=['lightsteelblue', 
#                    'cornflowerblue', 
#                    'royalblue', 
#                    'navy'])
#     plt.title(feat)
#     plt.savefig(os.path.join(experiment_dir, '{}.png'.format(feat)))
#     plt.clf()
#     if is_notebook:
#         plt.show()


# In[43]:


print(final_results_df)
final_results_df