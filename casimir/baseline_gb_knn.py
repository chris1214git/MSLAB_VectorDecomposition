import os
import sys
import numpy as np
import argparse
from collections import defaultdict

sys.path.append("../")

import torch
from torch.utils.data import random_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs, merge_targets
from load_pretrain_label import load_preprocess_document_labels

torch.set_num_threads(8)

def knn_evaluate(config, model, vocabulary, word_embeddings, X_test, y_test):
    results = defaultdict(list)
        
    # predict all data
    if config['target'] == 'yake':
        pred = torch.abs(torch.Tensor(model.predict(X_test)))
        y = torch.abs(torch.Tensor(y_test))
    else:
        pred = torch.Tensor(model.predict(X_test))
        y = torch.Tensor(y_test)
    # Semantic Prcision
    precision_scores, word_result = semantic_precision_all(pred, y, word_embeddings, vocabulary, k=config['topk'], th=config['threshold'])
    for k, v in precision_scores.items():
        results['Semantic Precision v1@{}'.format(k)].append(v)
    precision_scores, word_result = semantic_precision_all_v2(pred, y, word_embeddings, vocabulary, k=config['topk'], th=config['threshold'])
    for k, v in precision_scores.items():
        results['Semantic Precision_v2@{}'.format(k)].append(v)

    # Precision
    precision_scores = retrieval_precision_all(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision v1@{}'.format(k)].append(v)
    precision_scores = retrieval_precision_all_v2(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision v2@{}'.format(k)].append(v)
    
    # NDCG
    ndcg_scores = retrieval_normalized_dcg_all(pred, y, k=config["topk"])
    for k, v in ndcg_scores.items():
        results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results

def mean_evaluate(config, preds, labels, vocabulary, word_embeddings):
    results = defaultdict(list)
        
    # predict all data
    if config['target'] == 'yake':
        pred = torch.abs(torch.Tensor(preds))
        y = torch.abs(torch.Tensor(labels))
    else:
        pred = torch.Tensor(preds)
        y = torch.Tensor(labels)
    # Semantic Prcision
    precision_scores, word_result = semantic_precision_all(pred, y, word_embeddings, vocabulary, k=config['topk'], th=config['threshold'])
    for k, v in precision_scores.items():
        results['Semantic Precision v1@{}'.format(k)].append(v)
    precision_scores, word_result = semantic_precision_all_v2(pred, y, word_embeddings, vocabulary, k=config['topk'], th=config['threshold'])
    for k, v in precision_scores.items():
        results['Semantic Precision_v2@{}'.format(k)].append(v)

    # Precision
    precision_scores = retrieval_precision_all(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision v1@{}'.format(k)].append(v)
    precision_scores = retrieval_precision_all_v2(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision v2@{}'.format(k)].append(v)
    
    # NDCG
    ndcg_scores = retrieval_normalized_dcg_all(pred, y, k=config["topk"])
    for k, v in ndcg_scores.items():
        results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="KNN&Mean")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--dataset2', type=str, default=None)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--n_neighbors', type=int, default=20)
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
        print('[Cross Domain]')
        ### Dataset2
        # Parameter
        dataset=config['dataset']
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

        # generating document embedding
        doc_embs2, doc_model2, device2 = get_preprocess_document_embs(preprocessed_corpus2, config['encoder'])

        # merge two dataset
        targets1, targets2, new_vocabularys = merge_targets(label, label2, vocabularys, vocabularys2)

        # word embedding preparation
        word_embeddings = get_word_embs(new_vocabularys, data_type='tensor')

        # KNN
        model = KNeighborsRegressor(n_neighbors=config["n_neighbors"])
        model.fit(doc_embs, targets1)
        res = knn_evaluate(config, model, new_vocabularys, word_embeddings, doc_embs2, targets2)
        record = open('./'+'CrossDomain_'+dataset+'_'+config['dataset2']+'_KNN'+'_'+config['encoder']+'_'+config['target']+'.txt', 'a')
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
            record.write(f"{key}:{val:.4f}\n")
        
        # Mean
        predict = targets1.mean(axis=0)
        predict = np.tile(predict, (targets2.shape[0], 1))
        res = mean_evaluate(config, predict, targets2, new_vocabularys, word_embeddings)
        record = open('./'+'CrossDomain_'+dataset+'_'+config['dataset2']+'_MEAN'+'_'+config['encoder']+'_'+config['target']+'.txt', 'a')
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
            record.write(f"{key}:{val:.4f}\n")
    else:
        print('[Single Dataset]')
        # word embedding preparation
        word_embeddings = get_word_embs(vocabularys, data_type='tensor')

        # KNN
        X_train, X_test, y_train, y_test = train_test_split(doc_embs, label, test_size=0.2, random_state=config["seed"])
        model = KNeighborsRegressor(n_neighbors=config["n_neighbors"])
        model.fit(X_train, y_train)
        res = knn_evaluate(config, model, vocabularys, word_embeddings, X_test, y_test)
        record = open('./'+'baseline'+'_'+config['dataset']+'_KNN'+'_'+config['encoder']+'_'+config['target']+'.txt', 'a')
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
            record.write(f"{key}:{val:.4f}\n")

        # Mean
        predict = label.mean(axis=0)
        predict = np.tile(predict, (label.shape[0], 1))

        res = mean_evaluate(config, predict, label, vocabularys, word_embeddings)
        record = open('./'+'baseline'+'_'+config['dataset']+'_MEAN'+'_'+config['encoder']+'_'+config['target']+'.txt', 'a')
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
            record.write(f"{key}:{val:.4f}\n")