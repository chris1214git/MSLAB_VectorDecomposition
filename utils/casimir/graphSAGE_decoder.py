import os
import re
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_cluster import random_walk
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
#from collections import defaultdict

sys.path.append("../")
from model.graph_sage import GraphSAGE, GraphSAGE_Dataset
from utils.loss import ListNet, MythNet
from utils.data_processing import get_process_data
from utils.data_loader import load_document
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import get_free_gpu, same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs, split_data, doc_filter, generate_graph

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(8)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="GraphSAGE")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--use_pos', type=bool, default=False)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='bert')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--visualize', type=bool, default=True)
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
    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)

    # generating document embedding
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
    id2token = {k: v for k, v in zip(range(0, len(vocabularys[config['target']])), vocabularys[config['target']])}
    token2id = {v: k for k, v in id2token.items()}

    # word embedding preparation
    word_embeddings = get_word_embs(vocabularys[config['target']], id2token=id2token, data_type='tensor')

    # Show Setting
    config['vocab_size'] = len(vocabularys[config['target']])
    show_settings(config)
    record_settings(config)
    
    # Build Graph
    if config['model'] == 'GraphSAGE':
        vocab_set = set(token2id)
        doc_list = [doc_filter(doc, vocab_set) for doc in tqdm(preprocessed_corpus, desc="Delete word from preprocessed corpus:")]
        edge_index = torch.tensor(generate_graph(doc_list, token2id, id2token), dtype=torch.long).t().contiguous()
    else:
        edge_index = None

    # prepare dataset
    dataset = GraphSAGE_Dataset(unpreprocessed_corpus, doc_embs, labels[config['target']])
    training_length = int(len(dataset) * config['ratio'])
    validation_length = len(dataset) - training_length
    training_set, validation_set = random_split(dataset, lengths=[training_length, validation_length],generator=torch.Generator().manual_seed(42))

    # Define document embeddings dimension
    if config['encoder'] == 'doc2vec':
        contextual_size = 200
    elif config['encoder'] == 'average':
        contextual_size = 300
    else:
        contextual_size = 768

    # Declare model & train
    while True:
        try:
            model = GraphSAGE(config=config, edge_index=edge_index, vocabulary=vocabularys[config['target']], id2token=id2token, contextual_size=contextual_size, vocab_size=len(vocabularys[config['target']]), word_embeddings=word_embeddings)
            model.fit(training_set, validation_set)
            break
        except:
            print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)