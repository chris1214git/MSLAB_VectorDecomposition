import os
import sys
import math
import nltk
import time
import torch
import random
import argparse
import numpy as np

sys.path.append("../..")
from load_pretrain_label import load_preprocess_document_labels
from model.ide_gan_decoder import IDEDataset, IDEGanDecoder
from model.ide_bert_gan_decoder import IDEBERTGanDecoder
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_preprocess_document_labels_gensim_dct, get_word_embs, merge_targets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(15)

def generate_dataset(config):
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
    if config['model'] == 'IDE_BERT_GAN':
        labels, vocabularys, gensim_dct = get_preprocess_document_labels_gensim_dct(preprocessed_corpus)
        label = labels[config['target']]
        vocabularys = vocabularys[config['target']]
    else:
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
    
    idx = np.arange(len(unpreprocessed_corpus))
    np.random.shuffle(idx)
    train_length = int(len(unpreprocessed_corpus) * 0.8)
    train_idx = idx[:train_length]
    valid_idx = idx[train_length:]

    train_unpreprocessed_corpus = list(np.array(unpreprocessed_corpus)[train_idx])
    valid_unpreprocessed_corpus = list(np.array(unpreprocessed_corpus)[valid_idx])
    train_embs = np.array(doc_embs)[train_idx]
    valid_embs = np.array(doc_embs)[valid_idx]
    train_label = np.array(label)[train_idx]
    valid_label = np.array(label)[valid_idx]
    
    # Generate labeled mask
    label_masks = np.zeros((train_embs.shape[0], 1), dtype=bool)
    num_labeled_data = int(train_embs.shape[0] * config['ratio'])
    while True:
        if num_labeled_data > 0:
            idx = random.randrange(0, train_embs.shape[0])
            if label_masks[idx] == 0:
                label_masks[idx] = 1
                num_labeled_data -= 1
        else:
            break
    print('[INFO] mask labels done.')

    # Balance data if required
    original_num_data = train_embs.shape[0]
    if config['ratio'] != 1 and config['balance']:
        print('[INFO] Balance required.')
        for idx in range(original_num_data): 
            if label_masks[idx]:
                balance = int(1/config['ratio'])
                balance = int(math.log(balance,2))
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    train_unpreprocessed_corpus.append(train_unpreprocessed_corpus[idx])
                    train_embs = np.concatenate((train_embs, train_embs[idx].reshape(1, train_embs.shape[1])), axis=0)
                    train_label = np.concatenate((train_label, train_label[idx].reshape(1, train_label.shape[1])), axis=0)
                    label_masks = np.concatenate((label_masks, label_masks[idx].reshape(1, label_masks.shape[1])), axis=0)
    
    training_set = IDEDataset(train_unpreprocessed_corpus, train_embs, train_label, label_masks)
    validation_set = IDEDataset(valid_unpreprocessed_corpus, valid_embs, valid_label, np.ones((valid_embs.shape[0], 1), dtype=bool))
    
    if config['model'] == 'IDE_BERT_GAN':
        return training_set, validation_set, vocabularys, id2token, gensim_dct, device 
    else:
        return training_set, validation_set, vocabularys, id2token, device 

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--experiment', type=str, default="check")
    parser.add_argument('--model', type=str, default="IDE_GAN")
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
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='listnet')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--warmup', type=str, default='linear')
    parser.add_argument('--warmup_proportion', type=str, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ratio', type=float, default=1)
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
    
    # Generate dataset
    if config['model'] == 'IDE_BERT_GAN':
        training_set, validation_set, vocabularys, id2token, gensim_dct, device = generate_dataset(config)
        model = IDEBERTGanDecoder(config, training_set, validation_set, vocabularys, id2token, gensim_dct, device)
    elif config['model'] == 'IDE_GAN':
        training_set, validation_set, vocabularys, id2token, device = generate_dataset(config)
        model = IDEGanDecoder(config, training_set, validation_set, vocabularys, id2token, device)
    model.fit()