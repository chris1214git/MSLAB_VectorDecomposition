import os
import sys
import torch
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

sys.path.append("../")
from model.mlp_decoder import MLP, MLPDataset
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs, merge_targets
from load_pretrain_label import load_preprocess_document_labels

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--dataset2', type=str, default=None)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='listnet')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    config = vars(args)
    same_seeds(config["seed"])

    # Parameter
    if config['dataset'] == '20news':
        config['min_df'], config['max_df'], config['min_doc_word'] = 62, 1.0, 15
    elif config['dataset'] == 'agnews':
        config['min_df'], config['max_df'], config['min_doc_word'] = 425, 1.0, 15
    elif config['dataset'] == 'IMDB':
        config['min_df'], config['max_df'], config['min_doc_word'] = 166, 1.0, 15
    elif config['dataset'] == 'wiki':
        config['min_df'], config['max_df'], config['min_doc_word'] = 2872, 1.0, 15
    elif config['dataset'] == 'tweet':
        config['min_df'], config['max_df'], config['min_doc_word'] = 5, 1.0, 15
    
    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # Decode target & Vocabulary
    if config['target'] == 'keybert' or config['target'] == 'yake':
        labels, vocabularys= load_preprocess_document_labels(config)
        label = labels[config['target']].toarray()
    else:
        labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
        label = labels[config['target']]
        vocabularys = vocabularys[config['target']]

    # generating document embedding
    doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    # word embedding preparation
    word_embeddings = get_word_embs(vocabularys, data_type='tensor')

    # prepare dataset
    dataset = MLPDataset(doc_embs, label)
    training_length = int(len(dataset) * config['ratio'])
    validation_length = len(dataset) - training_length
    training_set, validation_set = random_split(dataset, lengths=[training_length, validation_length],generator=torch.Generator().manual_seed(42))
    
    # Declare model & train
    while True:
        try:
            model = MLP(config=config, vocabulary=vocabularys, contextual_size=doc_embs.shape[1], word_embeddings=word_embeddings)
            model.fit(training_set, validation_set)
            break
        except:
            print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)
    # model = MLP(config=config, vocabulary=vocabularys, contextual_size=doc_embs.shape[1], word_embeddings=word_embeddings)
    # model.fit(training_set, validation_set)
