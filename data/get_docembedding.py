import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_document(dataset):
    if dataset == "20news":
        num_classes = 20
        raw_text, target = fetch_20newsgroups(data_home="./", subset='all', categories=None,
                                shuffle=True, random_state=42, return_X_y=True)
        documents = [doc.strip("\n") for doc in raw_text]
    elif dataset == "IMDB":
        num_classes = 2
        documents = []
        target = []

        dir_prefix = "./aclImdb/train/"
        sub_dir = ["pos","neg"]
        for target_type in sub_dir:
            data_dir = os.path.join(dir_prefix, target_type)
            files_name = os.listdir(data_dir)
            for f_name in files_name:
                with open(os.path.join(data_dir,f_name),"r") as f:
                    context = f.readlines()
                    documents.extend(context)

            # assign label
            label = 1 if target_type=="pos" else 0
            label = [label]* len(files_name)
            target.extend(label)
    else:
        raise NotImplementedError
    
    return documents, target, num_classes


class Vocabulary:
    def __init__(self, min_word_freq_threshold=0, topk_word_freq_threshold=0):
        # The low frequency words will be assigned as <UNK> token
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}
        
        self.min_word_freq_threshold = min_word_freq_threshold
        self.topk_word_freq_threshold = topk_word_freq_threshold
        
        self.word_freq_in_corpus = defaultdict(int)
        self.IDF = {}
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.strip().split()
        
        return [self.ps.stem(w) for w in text if w.lower() not in self.stop_words]

    def build_vocabulary(self, sentence_list):
        self.word_vectors = []
        self.doc_freq = defaultdict(int) # # of document a word appear
        self.document_num = len(sentence_list)
        
        for sentence in tqdm(sentence_list, desc="Preprocessing documents"):
            # for doc_freq
            document_words = set()
            
            for word in self.tokenizer_eng(sentence):
                # calculate word freq
                self.word_freq_in_corpus[word] += 1
                document_words.add(word)
                
            for word in document_words:
                self.doc_freq[word] += 1
        
        # calculate IDF
        print('doc num', self.document_num)
        for word, freq in self.doc_freq.items():
            self.IDF[word] = math.log(self.document_num / (freq+1))
        
        # delete less freq words:
        delete_words = []
        for word, v in self.word_freq_in_corpus.items():
            if v < self.min_word_freq_threshold:
                delete_words.append(word)     
        for word in delete_words:
            del self.IDF[word]    
            del self.word_freq_in_corpus[word]
        
        # delete too freq words
        print('eliminate freq words')
        IDF = [(word, freq) for word, freq in self.IDF.items()]
        IDF.sort(key=lambda x: x[1])

        for i in range(self.topk_word_freq_threshold):
            word = IDF[i][0]
            del self.IDF[word]
            del self.word_freq_in_corpus[word]
        
        # construct word_vectors
        idx = 1
        for word in self.word_freq_in_corpus:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
            
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] for token in tokenized_text if token in self.stoi
        ]

class LSTM(nn.Module):
    def __init__(self, vocab_size,hidden_dim, output_dim,dropout_rate,):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = pack_padded_sequence(embedded, length, batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction

    def get_docvec(self,ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = pack_padded_sequence(embedded, length, batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        return hidden[-1]

def evaluate(model, test_loader, device):
    accuracy = []
    for word,length, target in test_loader:
        word, target = word.to(device), target.to(device)
        with torch.no_grad():
            logits = model(word,length)
        hits = logits.argmax(dim=1).eq(target)
        accuracy.append(hits)
    return torch.cat(accuracy).float().mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_word_freq_threshold', type=int, default=5)
    parser.add_argument('--topk_word_freq_threshold', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--max_seq_length', type=int, default=64)

    args = parser.parse_args()
    config = vars(args)

    # load document
    # print(f"Setting seeds:{config['seed']}")
    same_seeds(config["seed"])
    documents, target, num_classes = load_document(config["dataset"])
    max_seq_length = config["max_seq_length"]

    # build vocabulary
    vocab = Vocabulary(min_word_freq_threshold=config["min_word_freq_threshold"], 
                       topk_word_freq_threshold=config["topk_word_freq_threshold"])
    vocab.build_vocabulary(documents)
    print(f"Vocab size:{len(vocab)}")
    tokenize_data = []
    valid_label = []
    for sen_id, sen in enumerate(tqdm(documents, desc="Numericalizing")):
        numerical_output = vocab.numericalize(sen)[:max_seq_length]
        
        # some document becomes empty after filtering word
        if len(numerical_output) > 0:
            tokenize_data.append(torch.LongTensor(numerical_output))
            valid_label.append(target[sen_id])

    # prepare pytorch input
    # tokenize_data = sorted(tokenize_data, key = lambda x: len(x), reverse = True)
    seq_length = torch.IntTensor([len(i) for i in tokenize_data])
    paded_context = pad_sequence(tokenize_data,batch_first=True,padding_value=0)
    target_tensor = torch.LongTensor(valid_label)

    # dataset
    dataset = TensorDataset(paded_context, seq_length, target_tensor)
    train_length = int(len(dataset)*0.8)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = random_split(dataset,lengths=[train_length,valid_length])
    train_loader = DataLoader(train_dataset,batch_size = 128)
    valid_loader = DataLoader(valid_dataset,batch_size = 128)
    full_loader =  DataLoader(dataset,batch_size = 128)

    # training
    device = "cuda:0"
    model = LSTM(len(vocab), config["dim"],num_classes,0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(100):
        accuracy = []
        model.train()
        for word,length, target in tqdm(train_loader):
            word, target = word.to(device), target.to(device)
            
            logits = model(word,length)
            loss = loss_function(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            hits = logits.argmax(dim=1).eq(target)
            accuracy.append(hits)

        accuracy = torch.cat(accuracy).float().mean()
        print(f"[Epoch {epoch:02d}] Train Accuracy:{accuracy:.4f}")
        valid_acc = evaluate(model, valid_loader,device)
        print(f"[Epoch {epoch:02d}] Valid Accuracy:{valid_acc:.4f}")

    # save document embedding
    model.eval()
    document_representation = []
    for word,length, _ in tqdm(full_loader):
        word = word.to(device)
        with torch.no_grad():
            vectors = model.get_docvec(word,length)
            vectors = vectors.detach().cpu().numpy().tolist()
        document_representation.extend(vectors)

    print("Saving document vectors")
    np.save(f"docvec_20news_LSTM_{config['dim']}d.npy", document_representation)
        