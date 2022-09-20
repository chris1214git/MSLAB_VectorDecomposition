import os
import sys
import math
import nltk
import time
import torch
import random
import argparse
import numpy as np
from torch.utils.data import  random_split

sys.path.append("../")
from load_pretrain_label import load_preprocess_document_labels
from model.ide_da_decoder import IDEDataset, IDEDADecoder
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_preprocess_document_labels_gensim_dct, get_word_embs, merge_targets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(15)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--experiment', type=str, default="check")
    parser.add_argument('--model', type=str, default="IDE_GAN")
    parser.add_argument('--ae', type=str, default='no')
    parser.add_argument('--architecture', type=str, default="concatenate")
    parser.add_argument('--activation', type=str, default="sigmoid")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--dataset2', type=str, default=None)
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--target', type=str, default='tf-idf-gensim')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--de_epochs', type=int, default=500)
    parser.add_argument('--en_epochs', type=int, default=40)
    parser.add_argument('--ae_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='listnet')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--ae_lr', type=float, default=2e-3)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--warmup', type=str, default='linear')
    parser.add_argument('--warmup_proportion', type=str, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--check_document', type=bool, default=True)
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
    
    # Data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]
    print('[INFO] Load corpus done.')

    # Generating document embedding
    while True:
        try:
            doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])
            break
        except:
            print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)
    print('[INFO] Generate embedding done.')

    # Generate Decode target & Vocabulary
    if config['target'] == 'keybert' or config['target'] == 'yake':
        labels, vocabularys= load_preprocess_document_labels(config)
        label = labels[config['target']].toarray()
    else:
        labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
        label = labels[config['target']]
        vocabularys = vocabularys[config['target']]
    print('[INFO] Load label done.')

    # generate idx to token
    id2token = {k: v for k, v in zip(range(0, len(vocabularys)), vocabularys)}
    print('[INFO] Generate id2token done.')
    # real or fake
    reals = np.ones((doc_embs.shape[0], 1))
    
    dataset = IDEDataset(unpreprocessed_corpus, preprocessed_corpus, doc_embs, label, reals)
    label_length = int(len(dataset) * config['ratio'])
    unlabel_length = int(len(dataset) * (0.8 - config['ratio']))
    validation_length = len(dataset) - label_length - unlabel_length
    label_set, unlabel_set, validation_set = random_split(dataset, lengths=[label_length, unlabel_length, validation_length], generator=torch.Generator().manual_seed(42))

    model = IDEDADecoder(config, label_set, unlabel_set, validation_set, vocabularys, id2token, device)
    model.en_fit()
    model.de_fit()