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

import os
import sys
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

from sklearn.metrics import ndcg_score
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

train_ratio = 0.8
n_component = int(sys.argv[1])
document_vectors = np.load("document_vectors.npy", allow_pickle=True)

# Split training and testing set.
train_set = document_vectors[:int(train_ratio * len(document_vectors))]
test_set = document_vectors[int(train_ratio * len(document_vectors)):]

print("train set size:{}".format(train_set.shape))
print("test set size:{}".format(test_set.shape))

# Standardize features.
scaler = StandardScaler()

scaler.fit(train_set)

train_set = scaler.transform(train_set)

test_set = scaler.transform(test_set)

# Start PCA and reconstruct document vectors.

# pca = IncrementalPCA(n_components=n_component, batch_size=1000)
pca = PCA(n_components=n_component)


pca.fit(train_set)

# Test on test set and save reconstruct vectors.
document_embedding = pca.transform(test_set)

reconstruct_vectors = pca.inverse_transform(document_embedding)

print("Number of component:{}".format(n_component))
print("Original size:{}".format(test_set.shape))
print("Reconstruct size:{}".format(reconstruct_vectors.shape))
print("\n\n============================================")
np.save("reconstruct_{}.npy".format(n_component), reconstruct_vectors)