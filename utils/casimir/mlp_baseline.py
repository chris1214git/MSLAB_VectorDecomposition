import os
import sys
import time
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
torch.set_num_threads(15)

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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
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
    if config['dataset2'] is not None:
        print('[INFO] Cross Domain')
        dataset = config['dataset']
        config['dataset'] = config['dataset2']
        if config['dataset'] == '20news':
            config['min_df'], config['max_df'], config['min_doc_word'] = 62, 1.0, 15
        elif config['dataset'] == 'agnews':
            config['min_df'], config['max_df'], config['min_doc_word'] = 425, 1.0, 15
        elif config['dataset'] == 'IMDB':
            config['min_df'], config['max_df'], config['min_doc_word'] = 166, 1.0, 15
        elif config['dataset'] == 'wiki':
            config['min_df'], config['max_df'], config['min_doc_word'] = 2872, 1.0, 15
        elif config['dataset'] == 'tweet':
            config['min_df'], config['max_df'], config['min_doc_word'] = 1, 1.0, 1

        # data preprocessing
        unpreprocessed_corpus2 ,preprocessed_corpus2 = get_preprocess_document(**config)

        # Decode target & Vocabulary
        if config['target'] == 'keybert' or config['target'] == 'yake':
            labels2, vocabularys2= load_preprocess_document_labels(config)
            label2 = labels2[config['target']].toarray()
        else:
            labels2, vocabularys2= get_preprocess_document_labels(preprocessed_corpus2)
            label2 = labels2[config['target']]
            vocabularys2 = vocabularys2[config['target']]
        
        # Restore dataset
        config['dataset'] = dataset

        # generating document embedding
        doc_embs2, doc_model2, device2 = get_preprocess_document_embs(preprocessed_corpus2, config['encoder'])

        # merge two dataset
        targets1, targets2, new_vocabularys = merge_targets(label, label2, vocabularys, vocabularys2)

        # word embedding preparation
        word_embeddings = get_word_embs(new_vocabularys, data_type='tensor')

        # prepare dataset
        training_set = MLPDataset(doc_embs, targets1)
        validation_set = MLPDataset(doc_embs2, targets2)

        # Declare model & train
        while True:
            try:
                model = MLP(config=config, vocabulary=new_vocabularys, contextual_size=doc_embs.shape[1], word_embeddings=word_embeddings)
                model.fit(training_set, validation_set)
                break
            except:
                print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
                time.sleep(15)      
        # model = MLP(config=config, vocabulary=new_vocabularys, contextual_size=doc_embs.shape[1], word_embeddings=word_embeddings)
        # model.fit(training_set, validation_set)

    else:
        print('[INFO] Single Dataset')
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
