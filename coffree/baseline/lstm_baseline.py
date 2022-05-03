import sys
sys.path.append("../..")

import torch
import time
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn

from model import Decoder, Seq2Seq
from utils.toolbox import same_seeds, show_settings, get_preprocess_document, get_preprocess_document_embs, get_free_gpu

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

        doc_emb = batch.doc_emb
        trg = batch.trg

        optimizer.zero_grad()

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
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    word2idx, idx2word, labels = get_preprocess_document_labels(texts)

    device = get_free_gpu()

    vocabulary_size = len(word2idx)
    embedding_size = 512
    hidden_size = doc_embs.shape[1]
    num_layer = 1
    drop_out = 0

    print("doc_emb shape: {}".format(doc_embs.shape))
    print("voc size: {}".format(vocabulary_size))
    print("labels size: {}".format(labels.size()))

    # We only need decoder part
    # dec = Decoder(vocabulary_size, embedding_size, hidden_size, num_layer, drop_out)
    # model = Seq2Seq(dec, device).to(device)

    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])

    # N_EPOCHS = 10
    # CLIP = 1

    # for epoch in range(N_EPOCHS):
    #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    #     valid_loss = evaluate(model, valid_iterator, criterion)
