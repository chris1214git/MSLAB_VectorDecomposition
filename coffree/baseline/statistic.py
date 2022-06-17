import os
import sys
import numpy as np
import argparse
from collections import defaultdict

sys.path.append("../..")

import torch
from torch.utils.data import random_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs

torch.set_num_threads(8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="ZTM")
    parser.add_argument('--dataset', type=str, default="wiki")
    parser.add_argument('--target', type=str, default='tf-idf-gensim')
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--min_doc_len', type=int, default=15)
    parser.add_argument('--preprocess_config_dir', type=str, default='parameters_baseline2')
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--dataset2', type=str, default=None)
    args = parser.parse_args()
    config = vars(args)

    if config['dataset'] == '20news':
        config["min_df"], config['max_df'], config['min_doc_word'] = 62, 1.0, 15
    elif config['dataset'] == 'agnews':
        config["min_df"], config['max_df'], config['min_doc_word'] = 425, 1.0, 15
    elif config['dataset'] == 'IMDB':
        config["min_df"], config['max_df'], config['min_doc_word'] = 166, 1.0, 15
    elif config['dataset'] == 'wiki':
        config["min_df"], config['max_df'], config['min_doc_word'] = 2872, 1.0, 15

    show_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    unpreprocessed_corpus, preprocessed_corpus = get_preprocess_document(**config)

    # generating document embedding
    doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])
    print("Get doc embedding done.")
    
    texts = [text.split() for text in preprocessed_corpus]
    length = [len(text) for text in texts]
    print(texts[:3])

    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)

    print("Document number: {}\n".format(len(preprocessed_corpus)))
    print("Average doc len: {}\n".format(np.mean(length)))
    print("Vocabulary size: {}\n".format(len(vocabularys['tf-idf'])))