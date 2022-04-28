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
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--min_doc_len', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='bert')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    config = vars(args)
    config["dataset_name"] = config["dataset"]

    show_settings(config)
    record_settings(config)
    same_seeds(config["seed"])
    
    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]
    length = [len(text) for text in texts]
    print(texts[:3])

    # generating document embedding
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)

    print("Document number: {}\n".format(len(preprocessed_corpus)))
    print("Average doc len: {}\n".format(np.mean(length)))
    print("Vocabulary size: {}\n".format(len(vocabularys['tf-idf'])))