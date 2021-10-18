import argparse
import math
import os
import re
import numpy as np
from collections import defaultdict

import torch
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups


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
        self.doc_freq = defaultdict(int)  # of document a word appear
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

    def calculate_document_vector(self, sentence_list):
        # Calculate document representation (TFIDF).
        document_vectors = []
        word_dim = len(self.stoi)
        self.word_weight = self.IDF

        for sentence in tqdm(sentence_list, desc="calculate document vectors"):
            select_words = []
            document_vector = np.zeros(word_dim)
            for word in self.tokenizer_eng(sentence):
                # pass unknown word
                if word not in self.stoi:
                    continue
                else:
                    select_words.append(word)

            if len(select_words) == 0:
                print('error', sentence)
                continue

            # aggregate to doc vectors
            for word in select_words:
                # Here document vector will be TFIDF.
                document_vector[self.stoi[word]] += self.word_weight[word]

            document_vectors.append(document_vector)

        return document_vectors


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
        sub_dir = ["pos", "neg"]
        for target_type in sub_dir:
            data_dir = os.path.join(dir_prefix, target_type)
            files_name = os.listdir(data_dir)
            for f_name in files_name:
                with open(os.path.join(data_dir, f_name), "r") as f:
                    context = f.readlines()
                    documents.extend(context)

            # assign label
            label = 1 if target_type == "pos" else 0
            label = [label] * len(files_name)
            target.extend(label)
    else:
        raise NotImplementedError

    return documents, target, num_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_word_freq_threshold', type=int, default=5)
    parser.add_argument('--topk_word_freq_threshold', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()
    config = vars(args)

    # load document
    same_seeds(config["seed"])
    documents, target, num_classes = load_document(config["dataset"])

    # build vocabulary
    vocab = Vocabulary(min_word_freq_threshold=config["min_word_freq_threshold"],
                       topk_word_freq_threshold=config["topk_word_freq_threshold"])
    vocab.build_vocabulary(documents)
    print(f"Vocab size:{len(vocab)}")

    # Calculate TFIDF embedding.
    document_vectors = vocab.calculate_document_vector(documents)

    print(np.array(document_vectors).shape)

    np.save("document_vectors.npy", document_vectors)
