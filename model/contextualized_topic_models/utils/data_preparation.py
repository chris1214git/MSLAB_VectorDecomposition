import re
import torch
import numpy as np
import scipy.sparse
import warnings
import nltk
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from contextualized_topic_models.datasets.dataset import CTMDataset
from nltk.stem.snowball import EnglishStemmer
### casimir
# (1) import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.toolbox import get_free_gpu
###
from sklearn.preprocessing import OneHotEncoder
    
def load_word2emb(embedding_file):
    
    word2embedding = dict()
    word_dim = int(re.findall(r".(\d+)d", embedding_file)[0])

    with open(embedding_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            word2embedding[word] = np.array(embedding)

    print("Number of words:%d" % len(word2embedding))

    return word2embedding

def calculate_word_embeddings_tensor(word2embedding, tp):
    word_embeddings = torch.zeros(len(tp.vocab), len(word2embedding['a']))
    for k in tp.id2token:
        if tp.id2token[k] not in word2embedding:
            print('not found word embedding', tp.id2token[k])
            continue
        word_embeddings[k] = torch.tensor(word2embedding[tp.id2token[k]])

    return word_embeddings

def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=200):
    """
    Creates SBERT Embeddings from an input file
    """
    model = SentenceTransformer(sbert_model_to_load, device=get_free_gpu())
    with open(text_file, encoding="utf-8") as filino:
        train_text = list(map(lambda x: x, filino.readlines()))

    return np.array(model.encode(train_text, show_progress_bar=True, batch_size=batch_size))


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=200):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load, device=get_free_gpu())
    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))

def doc2vec_embeddings_from_list(texts):
    """
    Creates Doc2Vec Embeddings from an input file
    """
    embedding = []
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=200, workers=4)
    for idx in range(len(texts)):
        embedding.append(model.infer_vector(texts[idx]))
    return np.array(embedding)

def average_embeddings_from_list(texts, batch_size=200):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d', device=get_free_gpu())
    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))

class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None, show_warning=True, target='BoW', encoder='SBERT'):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None
        self.show_warning = show_warning
        self.target = target
        self.encoder = encoder
        self.stemmer = EnglishStemmer()
        self.tfidf_analyzer = TfidfVectorizer().build_analyzer()

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(contextualized_embeddings, bow_embeddings, id2token, labels)

    def stemmed_words(self, doc):
        return (self.stemmer.stem(w) for w in self.tfidf_analyzer(doc))

    def fit(self, text_for_contextual, text_for_bow, text_for_doc2vec, decode_target, vocab, id2token, labels=None, custom_embeddings=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None and custom_embeddings is None:
            raise Exception("A contextualized model or contextualized embeddings must be defined")

        if custom_embeddings and type(custom_embeddings).__module__ != 'numpy':
            raise TypeError("contextualized_embeddings must be a numpy.ndarray type object")

        ### casimir
        # if self.target == 'tfidf':   
        #     ## stemming verion  
        #     # self.vectorizer = TfidfVectorizer(analyzer=self.stemmed_words, token_pattern=r'(?u)\b[\w+|\-]+\b')
        #     ## w/o stemming version
        #     self.vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w+|\-]+\b')
        # elif self.target == 'bow_bin':
        #     self.vectorizer = CountVectorizer(binary=True)
        # else:
        #     self.vectorizer = CountVectorizer()
        # 
        # train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        train_bow_embeddings = decode_target
        ###
        if custom_embeddings is None:
            if self.encoder == 'doc2vec':
                ## doc2vec embedding
                train_contextualized_embeddings = doc2vec_embeddings_from_list(text_for_doc2vec)
            elif self.encoder == 'average':
                ## average word embedding
                train_contextualized_embeddings = average_embeddings_from_list(text_for_bow)
            else:
                ## SBERT embedding
                train_contextualized_embeddings = bert_embeddings_from_list(text_for_bow, self.contextualized_model)
        else:
            train_contextualized_embeddings = custom_embeddings
        ### casimir
        # comment
        # self.vocab = self.vectorizer.get_feature_names()
        # self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}
        self.vocab = vocab
        self.id2token = id2token
        ###

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        ### casimir
        # (1) Add one passing parameter (text_for_contexual). text_for_contextual is the list of raw documents
        return CTMDataset(train_contextualized_embeddings, train_bow_embeddings, text_for_contextual, self.id2token, encoded_labels)
        ###
    def transform(self, text_for_contextual, text_for_bow=None, custom_embeddings=None, labels=None):
        """
        This method create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM

        :param text_for_contextual: list of unpreprocessed documents to generate the contextualized embeddings
        :param text_for_bow: list of preprocessed documents for creating the bag-of-words
        :param custom_embeddings: np.ndarray type object to use custom embeddings (optional).
        :param labels: list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None:
            raise Exception("You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            if self.show_warning:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn("The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                          "are using ZeroShotTM in a cross-lingual setting")

            # we just need an object that is matrix-like so that pytorch does not complain
            test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(text_for_contextual), 1)))

        if custom_embeddings is None:
            test_contextualized_embeddings = bert_embeddings_from_list(text_for_contextual, self.contextualized_model)
        else:
            test_contextualized_embeddings = custom_embeddings

        if labels:
            encoded_labels = self.label_encoder.transform(np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(test_contextualized_embeddings, test_bow_embeddings, self.id2token, encoded_labels)
