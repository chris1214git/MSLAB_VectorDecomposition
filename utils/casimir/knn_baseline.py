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
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels_v3, get_word_embs

torch.set_num_threads(8)

def evaluate_Decoder(config, model, vocabulary, word_embeddings, X_test, y_test):
    results = defaultdict(list)
        
    # predict all data
    pred = torch.tensor(model.predict(X_test))
    y = torch.tensor(y_test)

    # Semantic Prcision
    precision_scores, word_result = semantic_precision_all(pred, y, word_embeddings, vocabulary, k=config['topk'], th=config['threshold'])
    for k, v in precision_scores.items():
        results['Semantic Precision@{}'.format(k)].append(v)

    # Precision
    precision_scores = retrieval_precision_all(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision@{}'.format(k)].append(v)
    
    # NDCG
    ndcg_scores = retrieval_normalized_dcg_all(pred, y, k=config["topk"])
    for k, v in ndcg_scores.items():
        results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="KNN")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='bert')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--topic_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--n_neighbors', type=int, default=20)
    args = parser.parse_args()
    
    config = vars(args)
    same_seeds(config["seed"])

    # Parameter
    if config['dataset'] == '20news':
        config['min_df'], config['max_df'], config['min_doc_word'] = 50, 1.0, 15
    elif config['dataset'] == 'agnews':
        config['min_df'], config['max_df'], config['min_doc_word'] = 100, 1.0, 15
    elif config['dataset'] == 'tweet':
        config['min_df'], config['max_df'], config['min_doc_word'] = 5, 1.0, 15
    
    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels_v3(config)
    id2token = {k: v for k, v in zip(range(0, len(vocabularys[config['target']])), vocabularys[config['target']])}

    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # generating document embedding
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    

    # word embedding preparation
    word_embeddings = get_word_embs(vocabularys[config['target']], id2token=id2token, data_type='tensor')

    # show setting
    config['vocab_size'] = len(vocabularys[config['target']])
    show_settings(config)
    record_settings(config)

    X_train, X_test, y_train, y_test = train_test_split(doc_embs, labels[config["target"]], test_size=0.2, random_state=config["seed"])

    X_train, X_test, y_train, y_test = train_test_split(doc_embs, labels[config["target"]], test_size=0.2, random_state=config["seed"])
    
    model = KNeighborsRegressor(n_neighbors=config["n_neighbors"])

    model.fit(X_train, y_train)

    res = evaluate_Decoder(config, model, vocabularys[config['target']], word_embeddings, X_test, y_test)

    record = open('./'+config['dataset']+'_'+config['model']+'_'+config['encoder']+'_'+config['target']+'.txt', 'a')
    for key,val in res.items():
        print(f"{key}:{val:.4f}")
        record.write(f"{key}:{val:.4f}\n")