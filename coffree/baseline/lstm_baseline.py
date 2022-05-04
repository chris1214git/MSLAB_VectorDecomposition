import sys
sys.path.append("../..")

import torch
import time
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

from model import Decoder, Seq2Seq
from torch.utils.data import DataLoader, Dataset
from utils.toolbox import same_seeds, show_settings, get_preprocess_document, \
                            get_preprocess_document_embs, get_free_gpu

class LSTMDecoderDataset(Dataset):
    def __init__(self, doc_embs, targets):
        
        assert len(doc_embs) == len(targets)

        self.doc_embs = torch.FloatTensor(doc_embs)
        self.targets = torch.LongTensor(targets)        
        # self.targets_rank = torch.argsort(self.targets, dim=1, descending=True)
        # self.topk = torch.sum(self.targets > 0, dim=1)
        
    def __getitem__(self, idx):
        return self.doc_embs[idx], self.targets[idx]

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

def prepare_dataloader(doc_embs, targets, batch_size=100, train_valid_test_ratio=[0.7, 0.1, 0.2]):
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
    
    # dataloader
    train_dataset = LSTMDecoderDataset(doc_embs[:train_size], targets[:train_size])
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = LSTMDecoderDataset(doc_embs[train_size:train_size+valid_size], targets[train_size:train_size+valid_size])
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_dataset = LSTMDecoderDataset(doc_embs[train_size+valid_size:], targets[train_size+valid_size:])
    test_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

def get_preprocess_document_labels(texts, max_len=50):
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


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        doc_emb, trg = batch
        trg = torch.transpose(trg, 0, 1)
        # doc_emb = [batch_size, emb_dim]
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output = model(doc_emb, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="ZTM")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--min_doc_len', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='bert')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    config = vars(args)

    show_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    unpreprocessed_corpus, preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # generating document embedding
    doc_embs, doc_model, device = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])
    print("Get doc embedding done.")

    word2idx, idx2word, labels = get_preprocess_document_labels(texts, max_len=config["max_len"])

    vocabulary_size = len(word2idx)
    embedding_size = 512
    hidden_size = doc_embs.shape[1]
    num_layer = 1
    drop_out = 0

    print("doc_emb shape: {}".format(doc_embs.shape))
    print("voc size: {}".format(vocabulary_size))
    print("labels size: {}".format(labels.size()))

    train_loader, valid_loader, test_loader = prepare_dataloader(doc_embs, labels, batch_size=32)

    # We only need decoder part
    dec = Decoder(vocabulary_size, embedding_size, hidden_size, num_layer, drop_out)
    model = Seq2Seq(dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])

    CLIP = 1

    # for epoch in range(config["num_epoch"]):
    #     train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    #     valid_loss = evaluate(model, valid_loader, criterion)
