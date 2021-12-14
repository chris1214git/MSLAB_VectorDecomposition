import math
import os
import re
import numpy as np
from collections import defaultdict

from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

import torch
from torch.nn.utils.rnn import pad_sequence

from .data_loader import load_document, load_word2emb


class Vocabulary:
    def __init__(self, document_list, agg="IDF", word2embedding=None,
                 min_word_freq_threshold=0, topk_word_freq_threshold=0, documentembedding_normalize=False):
        # The low frequency words will be assigned as <UNK> token
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}

        self.min_word_freq_threshold = min_word_freq_threshold
        self.topk_word_freq_threshold = topk_word_freq_threshold
        self.documentembedding_normalize = documentembedding_normalize
        
        self.IDF = {}
        self.agg = agg
        self.ps = PorterStemmer()
        self.document_list = document_list
        self.word2embedding = word2embedding
        self.word_freq_in_corpus = defaultdict(int)
        self.stop_words = set(stopwords.words('english'))

        self.build_vocabulary()

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.strip().split()

        return [self.ps.stem(w) for w in text if w.lower() not in self.stop_words]

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] for token in tokenized_text if token in self.stoi
        ]

    def init_word_weight(self):
        if self.agg == 'mean':
            self.word_weight = {word: 1 for word in self.IDF.keys()}
        elif self.agg == 'IDF':
            self.word_weight = self.IDF
        elif self.agg == 'uniform':
            self.word_weight = {word: np.random.uniform(
                low=0.0, high=1.0) for word in self.IDF.keys()}
        elif self.agg == 'gaussian':
            mu, sigma = 10, 1  # mean and standard deviation
            self.word_weight = {word: np.random.normal(
                mu, sigma) for word in self.IDF.keys()}
        elif self.agg == 'exponential':
            self.word_weight = {word: np.random.exponential(
                scale=1.0) for word in self.IDF.keys()}
        elif self.agg == 'pmi':
            trigram_measures = BigramAssocMeasures()
            self.word_weight = defaultdict(int)
            corpus = []

            for text in tqdm(self.document_list):
                corpus.extend(text.split())

            finder = BigramCollocationFinder.from_words(corpus)
            for pmi_score in finder.score_ngrams(trigram_measures.pmi):
                pair, score = pmi_score
                self.word_weight[pair[0]] += score
                self.word_weight[pair[1]] += score

    def document2index(self, dataset, max_seq_length=128):
        # Transform word to index.
        index_data = {}
        valid_label = []
        tokenize_data = []

        for sen_id, sen in enumerate(tqdm(dataset["documents"], desc="Numericalizing")):
            numerical_output = self.numericalize(sen)[:max_seq_length]

            # some document becomes empty after filtering word.
            if len(numerical_output) > 0:
                tokenize_data.append(torch.LongTensor(numerical_output))
                valid_label.append(dataset["target"][sen_id])

        index_data["seq_length"] = torch.IntTensor(
            [len(i) for i in tokenize_data])
        index_data["paded_context"] = pad_sequence(
            tokenize_data, batch_first=True, padding_value=0)
        index_data["target_tensor"] = torch.LongTensor(valid_label)

        return index_data

    def build_vocabulary(self):
        # Documents preprocessing.
        self.doc_freq = defaultdict(int)  # of document a word appear
        self.document_num = len(self.document_list)

        for sentence in tqdm(self.document_list, desc="Start buiding vocabulary..."):
            # for doc_freq
            document_words = set()

            for word in self.tokenizer_eng(sentence):
                # Pass unknow word if use pretrain embedding.
                if (self.word2embedding != None and word not in self.word2embedding):
                    continue

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

    def check_docemb(self):
        word_vectors = np.array(self.word_vectors)
        pred = np.zeros(word_vectors.shape[1])
        cnt = 0

        for word_idx in self.document_answers_idx[0]:
            pred += word_vectors[word_idx] * self.word_weight[self.itos[word_idx]]
            cnt += self.word_weight[self.itos[word_idx]]
        
        if self.config["document_vector_weight_normalize"]:
            pred /= cnt
        assert np.sum(self.document_vectors[0]) - np.sum(pred) == 0
        
    def get_document_representation(self):

        # Calculate document representation (TFIDF and weighted embedding).
        document_error = []
        document_word_weights = []
        document_embeddings = []

        word_dim = len(self.stoi)
        embedding_dim = len(self.word2embedding.get(
            "apple", 0)) if self.word2embedding != None else 0

        print("Vocabulary size:{}, Word embedding dim:{}".format(
            word_dim, embedding_dim))

        print("Initial word weight")
        self.init_word_weight()
        
        for sen_id, sentence in enumerate(tqdm(self.document_list, desc="Calculate document vectors...")):

            # Prepare document representation for each document.
            select_words = []
            document_word_weight = np.zeros(word_dim)
            document_embedding = np.zeros(embedding_dim)

            for word in self.tokenizer_eng(sentence):
                # pass unknown word
                if word not in self.stoi:
                    continue
                else:
                    select_words.append(word)
                    
            # aggregate doc vectors
            for word in select_words:
                document_word_weight[self.stoi[word]] += self.IDF[word]
                if (self.word2embedding != None):
                    document_embedding += self.word2embedding[word] * \
                        self.word_weight[word]

            if len(select_words) <= 5:
                document_error.append(sen_id)
                # print('error', sentence)
                # continue
            else:
                if self.documentembedding_normalize:
                    total_weight = np.sum(document_word_weight)
                    document_embedding /= total_weight
                    document_word_weight /= total_weight
                    
            document_word_weights.append(document_word_weight)
            document_embeddings.append(document_embedding)
        
        return document_word_weights, document_embeddings, document_error

def normalize_wordemb(word2embedding):
    # Every word emb should have norm 1

    word_emb = []
    word_list = []
    for word, emb in word2embedding.items():
        word_list.append(word)
        word_emb.append(emb)

    word_emb = np.array(word_emb)

    for i in range(len(word_emb)):
        norm = np.linalg.norm(word_emb[i])
        word_emb[i] = word_emb[i] / norm

    for word, emb in tqdm(zip(word_list, word_emb)):
        word2embedding[word] = emb
    return word2embedding


def del_error_documents(document_data, document_word_weight, document_embedding, document_error):
    print('delete items', len(document_error))
    for err_idx in sorted(document_error, reverse=True):
        # print('error', document_data["documents"][err_idx]) 
        # print('target', document_data["target"][err_idx])
        del document_data["documents"][err_idx]
        del document_data["target"][err_idx]
        del document_word_weight[err_idx]
        del document_embedding[err_idx]

    return document_data, document_word_weight, document_embedding


def get_process_data(dataset: str, agg: str = 'IDF', embedding_type: str = '', 
                     word2embedding_path: str = '../data/glove.6B.100d.txt',
                     word2embedding_normalize: bool = False, documentembedding_normalize: bool = False,
                     embedding_dim: int = 128, min_word_freq_threshold: int = 5,
                     topk_word_freq_threshold: int = 100, max_seq_length: int = 128,
                     load_embedding: bool = True) -> dict:
    # Input contents:
    # (1). dataset: Dataset name use for training and inference
    # (2). embedding_type: Return document embedding used for directly training decoder.
    # (3). word2embedding_path: Pretrain embedding model path, such as glove.6B.100d.txt.
    # (4). documentembedding_normalize: True for weighted mean, False for weighted sum 
    # Return contents:
    # (1). document_word_weight: Tfidf vectors for all documents, shape: [num_documents, vocab_size]
    # (2). document_embedding: Embedding vecotrs create from pretrain encoder, shape: [num_documents, embedding_dim]
    # (3). dataset: raw data, target and num_classes which used to train downstream task.
    # (4). LSTM_data: Contrains seq_length, paded_context, target_tensor, used to train LSTM autoencoder.
    
    word2embedding = None

    # Prepare dataset.
    document_data = load_document(dataset)

    if word2embedding_path != '':
        if not os.path.exists(word2embedding_path):
            print('Error: no word embedding', word2embedding_path)
            exit()

        print("Loading word2embedding from {}".format(word2embedding_path))
        word2embedding = load_word2emb(word2embedding_path)
        if word2embedding_normalize:
            print("Normalizing word2embedding")
            word2embedding = normalize_wordemb(word2embedding)

    vocab = Vocabulary(document_data["documents"], agg=agg, word2embedding=word2embedding, 
                       min_word_freq_threshold=min_word_freq_threshold, topk_word_freq_threshold=topk_word_freq_threshold,
                       documentembedding_normalize=documentembedding_normalize)

    # Prepare document representations.
    document_word_weight, document_embedding, document_error = vocab.get_document_representation()

    document_data, document_word_weight, document_embedding = del_error_documents(
        document_data, document_word_weight, document_embedding, document_error)

    document_data["target"] = np.array(document_data["target"])
    document_word_weight = np.array(document_word_weight)
    document_embedding = np.array(document_embedding)

    index_data = None
    # Prepare document embedding.
    if (embedding_type == "LSTM"):
        document_embedding = np.load("../data/docvec_20news_LSTM_{}d.npy".format(embedding_dim))
        index_data = vocab.document2index(document_data, max_seq_length)
    elif (embedding_type == "BERT"):
        document_embedding = np.load("docvec_20news_BertMLM.npy")
    
    return {"document_word_weight": document_word_weight, "document_embedding": document_embedding,
            "dataset": document_data, "LSTM_data": index_data}
