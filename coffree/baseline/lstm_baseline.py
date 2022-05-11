import os
import json
import sys
sys.path.append("../..")

import torch
import time
import random
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

from scipy import sparse
from model import Decoder, Seq2Seq
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2
from utils.toolbox import same_seeds, show_settings, get_preprocess_document, \
                            get_preprocess_document_embs, get_free_gpu, get_word_embs


class LSTMDecoderDataset(Dataset):
    def __init__(self, doc_embs, targets, labels):
        
        assert len(doc_embs) == len(targets)

        self.doc_embs = torch.FloatTensor(doc_embs)
        self.targets = torch.LongTensor(targets)      
        self.labels = torch.FloatTensor(labels)        # TFIDF
        # self.targets_rank = torch.argsort(self.targets, dim=1, descending=True)
        # self.topk = torch.sum(self.targets > 0, dim=1)
        
    def __getitem__(self, idx):
        return self.doc_embs[idx], self.targets[idx], self.labels[idx]

    def __len__(self):
        return len(self.doc_embs)

def pad_sequence(sentence, word2idx, sen_len):
    # 將每個句子變成一樣的長度
    if len(sentence) > sen_len:
        sentence = sentence[:sen_len]
    else:
        pad_len = sen_len - len(sentence)
        for _ in range(pad_len):
            sentence.append(word2idx["<PAD>"])
    assert len(sentence) == sen_len
    return sentence

def prepare_dataloader(doc_embs, targets, labels, batch_size=100, train_valid_test_ratio=[0.7, 0.1, 0.2]):
    train_size = int(len(doc_embs) * train_valid_test_ratio[0])
    valid_size = int(len(doc_embs) * (train_valid_test_ratio[0] + train_valid_test_ratio[1])) - train_size
    test_size = len(doc_embs) - train_size - valid_size
    
    print('Preparing dataloader')
    print('train size', train_size)
    print('valid size', valid_size)
    print('test size', test_size)

    # shuffle
    randomize = np.arange(len(doc_embs))
    np.random.shuffle(randomize)
    doc_embs = doc_embs[randomize]
    targets = targets[randomize]
    labels = labels[randomize]
    
    # dataloader
    train_dataset = LSTMDecoderDataset(doc_embs[:train_size], targets[:train_size], labels[:train_size])
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = LSTMDecoderDataset(doc_embs[train_size:train_size+valid_size], targets[train_size:train_size+valid_size], labels[train_size:train_size+valid_size])
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_dataset = LSTMDecoderDataset(doc_embs[train_size+valid_size:], targets[train_size+valid_size:], labels[train_size+valid_size:])
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

def get_document_labels(texts, max_len=50):
    word2idx = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>" : 3}
    idx2word = {0 : "<SOS>", 1 : "<EOS", 2 : "<PAD>", 3 : "<UNK>"}
    # Build dictionary
    for text in texts:
        for word in text:
            if (word2idx.get(word, -1) == -1):
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
    
    # Build labels
    # 把句子裡面的字轉成相對應的 index
    sentence_list = []
    for i, sen in enumerate(texts):
        sentence_idx = [word2idx["<SOS>"]]
        for word in sen:
            if (word in word2idx.keys()):
                sentence_idx.append(word2idx[word])
            else:
                sentence_idx.append(word2idx["<UNK>"])
        # 將每個句子變成一樣的長度
        sentence_idx = pad_sequence(sentence_idx, word2idx, max_len)
        sentence_idx[-1] = word2idx["<EOS>"]
        sentence_list.append(sentence_idx)

    labels = torch.LongTensor(sentence_list)
    return word2idx, idx2word, labels

def get_preprocess_document_labels_v2(preprocessed_corpus, config, preprocess_config_dir, ngram=1):
    print('Getting preprocess documents labels')
    print('Finding precompute_keyword by config', config)

    config_dir = os.path.join('../../data/precompute_keyword', preprocess_config_dir, \
                              '{}_ngram_{}'.format(config['dataset'], ngram))

    # Create tfidf target
    vectorizer = TfidfVectorizer()
    targets = vectorizer.fit_transform(preprocessed_corpus).toarray()

    bow_vector = sparse.load_npz(os.path.join(config_dir, 'BOW.npz'))
    try:
        keybert_vector = sparse.load_npz(os.path.join(config_dir, 'KeyBERT.npz'))
        yake_vector = sparse.load_npz(os.path.join(config_dir, 'YAKE.npz'))
    except:
        print('no precompute keyword')
        keybert_vector = None
        yake_vector = None

    if config["target"] == "tf-idf":
        vocabulary = vectorizer.get_feature_names()
    else:
        vocabulary = np.load(os.path.join(config_dir, 'vocabulary.npy'))

    labels = {}
    labels['tf-idf'] = targets
    labels['bow'] = bow_vector
    labels['keybert'] = keybert_vector
    labels['yake'] = yake_vector
    
    return labels, vocabulary


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        doc_emb, trg, _ = batch
        doc_emb = doc_emb.to(device)
        trg = torch.transpose(trg, 0, 1).to(device)
        # doc_emb = [batch_size, emb_dim]
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output = model(doc_emb, trg)

        output_dim = output.shape[-1]

        # print(trg.size())
        # print(output.size())

        trg = trg[1:].reshape(-1)
        output = output[1:].view(-1, output_dim)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_Decoder(model, data_loader, config, target_word2idx, vocab, word_embeddings):
    results = defaultdict(list)
    model.eval()
    
    # predict all data
    for data in data_loader:
        doc_embs, _, target = data
        
        doc_embs = doc_embs.to(device)
        target = target.to(device)
        _, pred = model.predict(doc_embs, word2idx, idx2word, target_word2idx)
        pred = pred.to(device)

        # Precision
        precision_scores = retrieval_precision_all(pred, target, k=config["topk"])
        for k, v in precision_scores.items():
            results['precision@{}'.format(k)].append(v)

        # Precision
        precision_scores = retrieval_precision_all_v2(pred, target, k=config["topk"])
        for k, v in precision_scores.items():
            results['precisionv2@{}'.format(k)].append(v)

        # NDCG
        ndcg_scores = retrieval_normalized_dcg_all(pred, target, k=config["topk"])
        for k, v in ndcg_scores.items():
            results['ndcg@{}'.format(k)].append(v)

        # Semantic Prcision for reconstruct
        precision_scores, word_result = semantic_precision_all(pred, target, word_embeddings, vocab, k=config['topk'], th = config['threshold'])
        for k, v in precision_scores.items():
            results['Semantic Precision v1@{}'.format(k)].append(v)

        precision_scores, word_result = semantic_precision_all_v2(pred, target, word_embeddings, vocab, k=config['topk'], th = config['threshold'])
        for k, v in precision_scores.items():
            results['Semantic Precision v2@{}'.format(k)].append(v)

    for k in results:
        results[k] = np.mean(results[k])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="ZTM")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--min_doc_len', type=int, default=15)
    parser.add_argument('--preprocess_config_dir', type=str, default='parameters_baseline2')
    parser.add_argument('--encoder', type=str, default='mpnet')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--min_df', type=float, default=0.003)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    config = vars(args)

    if config['dataset'] == '20news':
        config['max_df'], config['min_doc_word'] = 1.0, 15
    elif config['dataset'] == 'agnews':
        config['max_df'], config['min_doc_word'] = 1.0, 15
    elif config['dataset'] == 'IMDB':
        config['max_df'], config['min_doc_word'] = 1.0, 15
    elif config['dataset'] == 'wiki':
        config['max_df'], config['min_doc_word'] = 1.0, 15
    elif config['dataset'] == 'tweet':
        config['max_df'], config['min_doc_word'] = 1.0, 15

    show_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    unpreprocessed_corpus, preprocessed_corpus = get_preprocess_document(**config)

    preprocessed_corpus = preprocessed_corpus

    texts = [text.split() for text in preprocessed_corpus]

    word2idx, idx2word, labels = get_document_labels(texts, max_len=config["max_len"])

    # generating document embedding
    doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])
    print("Get doc embedding done.")

    label, vocabulary = get_preprocess_document_labels_v2(preprocessed_corpus, config, config['preprocess_config_dir'])
    targets = label[config["target"]].toarray()
    target_word2idx = {}
    for idx, word in enumerate(vocabulary):
        target_word2idx[word] = idx

    word_embeddings = get_word_embs(vocabulary, data_type='tensor', word_emb_file='../../data/glove.6B.300d.txt')

    vocabulary_size = len(word2idx)
    embedding_size = 512
    hidden_size = doc_embs.shape[1]
    num_layer = 1
    drop_out = 0

    print("doc_emb shape: {}".format(doc_embs.shape))
    print("voc size: {}".format(vocabulary_size))
    print("labels size: {}".format(labels.size()))

    train_loader, valid_loader, test_loader = prepare_dataloader(doc_embs, labels, targets, batch_size=32)

    # We only need decoder part
    dec = Decoder(vocabulary_size, embedding_size, hidden_size, num_layer, drop_out)
    model = Seq2Seq(dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])

    CLIP = 1

    # Start training
    for epoch in range(config["num_epoch"]):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        if ((epoch + 1) % 10 == 0): 
            print("Epoch:{}/{}, train_loss:{}".format(epoch+1, config["num_epoch"], train_loss))

    res = evaluate_Decoder(model, test_loader, config, target_word2idx, vocabulary, word_embeddings)
    for key, val in res.items():
        print(f"{key}:{val:.4f}")