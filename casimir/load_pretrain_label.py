import os
import re
import sys
import nltk
import torch
import numpy as np
import json
from math import log
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch.utils.data import DataLoader, random_split
from scipy import sparse

sys.path.append("../")
from utils.preprocessing import WhiteSpacePreprocessing, WhiteSpacePreprocessingStopwords, WhiteSpacePreprocessing_v2
from utils.data_loader import load_document, load_word2emb
from sklearn.feature_extraction.text import TfidfVectorizer

def load_preprocess_document_labels(config):
    print('Load preprocess documents labels')
    ngram = 1
    config_dir = os.path.join('../data/parameters_baseline2',
                              '{}_ngram_{}'.format(config['dataset'], ngram))
    # check preprocess config the same when loading precompute labels
    # with open(os.path.join(config_dir, 'preprocess_config.json'), 'r') as f:
    #     preprocess_config2 = json.load(f)
    #     assert preprocess_config == preprocess_config2

    tf_idf_vector = sparse.load_npz(os.path.join(config_dir, 'TFIDF.npz'))
    bow_vector = sparse.load_npz(os.path.join(config_dir, 'BOW.npz'))
    try:
        keybert_vector = sparse.load_npz(os.path.join(config_dir, 'KeyBERT.npz'))
        yake_vector = sparse.load_npz(os.path.join(config_dir, 'YAKE.npz'))
    except:
        print('no precompute keyword')
        keybert_vector = None
        yake_vector = None

    vocabulary = np.load(os.path.join(config_dir, 'vocabulary.npy'))

    labels = {}
    labels['tf-idf'] = tf_idf_vector
    labels['bow'] = bow_vector
    labels['keybert'] = keybert_vector
    labels['yake'] = yake_vector
    
    return labels, vocabulary