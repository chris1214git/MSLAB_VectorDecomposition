import os
import re
import sys
import nltk
import torch
import numpy as np
from math import log
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch.utils.data import DataLoader, random_split

sys.path.append("../")
from utils.preprocessing import WhiteSpacePreprocessing, WhiteSpacePreprocessingStopwords, WhiteSpacePreprocessing_v2
from utils.data_loader import load_document, load_word2emb
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_document(raw_documents):
    sp = WhiteSpacePreprocessingStopwords(raw_documents, stopwords_list=['english'], vocabulary_size=10000, min_words=15)
    preprocessed_documents, unpreprocessed_corpus, vocab, _ = sp.preprocess()
    delete_non_eng_documents = delete_non_eng(preprocessed_documents)
    noun_documents = pos(delete_non_eng_documents)
    delete_documents = []
    for idx in range(len(noun_documents)):
        if len(noun_documents[idx]) == 0:
            delete_documents.append(idx)
    delete_documents = sorted(delete_documents, reverse=True)
    for idx in delete_documents:
        del unpreprocessed_corpus[idx]
    noun_documents = list(filter(None, noun_documents))
    texts = [text.split() for text in noun_documents]
    return noun_documents, unpreprocessed_corpus, texts, vocab

def generate_document_embedding(model, documents):
    if model == 'roberta':
        model = SentenceTransformer("paraphrase-distilroberta-base-v1", device=get_free_gpu())
    else:
        model = SentenceTransformer("all-mpnet-base-v2", device=get_free_gpu())

    return np.array(model.encode(documents, show_progress_bar=True, batch_size=200))

def tokenizer_eng(text):
        text = re.sub(r'[^A-Za-z ]+', '', text)
        text = text.strip().split()
        return text

def delete_non_eng(documents):
    preprocessed_documents = []
    for text in documents:
        selected_word = []
        for word in tokenizer_eng(text):
            selected_word.append(word)
        preprocessed_documents.append(" ".join(selected_word))
    return preprocessed_documents


def pos(documents):
    is_noun = lambda pos: pos[:2] == 'NN'
    is_verb = lambda pos: pos[:2] == 'VB'

    preprocessed_documents = []
    for text in documents:
        tokenized = nltk.word_tokenize(text)
        noun_word = []
        for (word, pos) in nltk.pos_tag(tokenized):
            if is_noun(pos) or is_verb(pos):
                noun_word.append(word)
        preprocessed_documents.append(" ".join(noun_word))
    return preprocessed_documents

def calculate_word_embeddings_tensor(word2embedding, vocab, idx2token):
    word_embeddings = torch.zeros(len(vocab), len(word2embedding['a']))
    for k in idx2token:
        if idx2token[k] not in word2embedding:
            # print('not found word embedding', idx2token[k])
            continue
        word_embeddings[k] = torch.tensor(word2embedding[idx2token[k]])

    return word_embeddings

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    os.system('rm -f tmp')
    print('Using cuda {} for training...'.format(int(np.argmax(memory_available))))
    torch.cuda.device(int(np.argmax(memory_available)))
    return "cuda:{}".format(int(np.argmax(memory_available)))


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_settings(config):
    print('-------- Info ---------')
    settings = ""
    for key in list(config.keys()):
        settings += "{}: {}\n".format(key, config.get(key))
    print(settings)
    print('-----------------------')

def record_settings(config):
    record = open('./'+config['model']+'_'+config['dataset']+'_'+config['target']+'.txt', 'a')
    record.write('-------- Info ---------\n')
    settings = ""
    for key in list(config.keys()):
        settings += "{}: {}\n".format(key, config.get(key))
    record.write(settings)
    record.write('-----------------------\n')

def split_data(dataset, config):
    train_length = int(len(dataset)*0.6)
    valid_length = int(len(dataset)*0.2)
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, lengths=[train_length, valid_length, test_length],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, valid_loader, test_loader

def doc_filter(raw_document, vocab):
    PATTERN = r"(?u)\b\w\w+\b"
    doc = re.findall(PATTERN, raw_document.lower())
    return [x for x in doc if x in vocab]

def generate_graph(doc_list, word2index, index2word):
    window_size = 10
    windows = []

    # Traverse Each Document & Move window on each of them
    for doc in doc_list:
        length = len(doc)
        if length <= window_size:
            windows.append(doc)
        else:
            for i in range(length-window_size+1):
                window = doc[i: i+window_size]
                windows.append(window)
    
    word_freq = {}
    word_pair_count = {}
    for window in tqdm(windows, desc='Calculate word pair: '):
        appeared = set()
        for i in range(len(window)):
            if window[i] not in appeared:
                if window[i] in word_freq:
                    word_freq[window[i]] += 1
                else:
                    word_freq[window[i]] = 1
                appeared.add(window[i])
            if i != 0:
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word2index[word_i]
                    word_j = window[j]
                    word_j_id = word2index[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
    
    row = []
    col = []
    edge = []
    weight = []
    # pmi as weights

    num_window = len(windows)
    # count_mean = np.array(list(word_pair_count.values())).mean()
    for key in tqdm(word_pair_count, desc='Construct Edge: '):
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_freq[index2word[i]]
        word_freq_j = word_freq[index2word[j]]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        if count >= 15:
            edge.append([i, j])
            edge.append([j, i])
        weight.append(pmi)

    print('# of Node: {}\n# of Edge: {}'.format(len(word2index), len(edge)))

    return edge

def get_preprocess_document(dataset, min_df=1, max_df=1.0, vocabulary_size=None, min_doc_word=15, **kwargs):
    '''
    Returns preprocessed_docs & unpreprocessed_docs of the dataset

            Parameters:
                    dataset (str): For data_loader
                    min_df, max_df, vocabulary_size: For CountVectorizer in CTM preprocess
                    min_doc_word: Minimum doc length
            Returns:
                    unpreprocessed_docs (list):
                    preprocessed_docs (list):
    '''
    print('Getting preprocess documents:', dataset)
    print(f'min_df: {min_df} max_df: {max_df} vocabulary_size: {vocabulary_size} min_doc_word: {min_doc_word}')
    raw_documents = load_document(dataset)["documents"]
    # CTM preprocess
    sp = WhiteSpacePreprocessing_v2(raw_documents, stopwords_language='english',\
                                    min_df=min_df, max_df=max_df, vocabulary_size=vocabulary_size)

    preprocessed_docs, unpreprocessed_docs, vocabulary, _ = sp.preprocess()
    # filter special character
    preprocessed_docs = delete_non_eng(preprocessed_docs)
    # select nouns & verbs
    preprocessed_docs = pos(preprocessed_docs)
    # delete short articles
    delete_docs_idx = []
    for idx in range(len(preprocessed_docs)):
        # length > min_doc_word
        if len(preprocessed_docs[idx]) == 0 or len(preprocessed_docs[idx]) < min_doc_word:
            delete_docs_idx.append(idx)
    delete_docs_idx = sorted(delete_docs_idx, reverse=True)
    for idx in delete_docs_idx:
        del preprocessed_docs[idx]
        del unpreprocessed_docs[idx]
    
    return unpreprocessed_docs ,preprocessed_docs

def get_preprocess_document_labels(preprocessed_docs):
    '''
    Returns labels for document decoder

            Parameters:
                    preprocessed_docs (list): 
            Returns:
                    labels (dict): bow, tf-idf
                    vocabulary (dict): bow, tf-idf
    '''
    print('Getting preprocess documents labels')
    vectorizer = TfidfVectorizer()
    # covert sparse matrix to numpy array
    tf_idf_vector = vectorizer.fit_transform(preprocessed_docs).toarray()
    bow_vector = tf_idf_vector.copy()
    bow_vector[bow_vector > 0] = 1
    bow_vector[bow_vector < 0] = 0
    vocabulary = vectorizer.get_feature_names()

    id2token = {k: v for k, v in zip(range(0, len(vocabulary)), vocabulary)}

    labels = {}
    labels['tf-idf'] = tf_idf_vector
    labels['bow'] = bow_vector
    
    vocabularys = {}
    vocabularys['tf-idf'] = vocabulary
    vocabularys['bow'] = vocabulary

    

    return labels, vocabularys, id2token

def get_preprocess_document_embs(preprocessed_docs, model_name):
    '''
    Returns embeddings(input) for document decoder

            Parameters:
                    preprocessed_docs (list): 
                    model_name (str):
            Returns:
                    doc_embs (array): 
                    model (class): 
    '''
    print('Getting preprocess documents embeddings')
    if model_name == 'roberta':
        model = SentenceTransformer("paraphrase-distilroberta-base-v1", device=get_free_gpu())
        doc_embs = np.array(model.encode(preprocessed_docs, show_progress_bar=True, batch_size=200))
    elif model_name == 'mpnet':
        model = SentenceTransformer("all-mpnet-base-v2", device=get_free_gpu())
        doc_embs = np.array(model.encode(preprocessed_docs, show_progress_bar=True, batch_size=200))
    elif model_name == 'average':
        model = SentenceTransformer("average_word_embeddings_glove.840B.300d", device=get_free_gpu())
        doc_embs = np.array(model.encode(preprocessed_docs, show_progress_bar=True, batch_size=200))
    elif model_name == 'doc2vec':
        doc_embs = []
        preprocessed_docs_split = [doc.split() for doc in preprocessed_docs]
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocessed_docs_split)]
        model = Doc2Vec(documents, vector_size=200, workers=4)
        for idx in range(len(preprocessed_docs_split)):
            doc_embs.append(model.infer_vector(preprocessed_docs_split[idx]))
        doc_embs = np.array(doc_embs)

    return doc_embs, model   

def get_word_embs(vocabularys, word_emb_file='../data/glove.6B.300d.txt'):
    '''
    Returns word_embs array for semantic precision

            Parameters:
                    vocabularys (list): 
                    word_emb_file (str): 
            Returns:
                    word_embs (array): 
    '''
    print('Getting word embeddings')
    word2emb = load_word2emb(word_emb_file)
    dim = len(list(word2emb.values())[0])

    word_embs = []
    for word in vocabularys:
        if word not in word2emb:
            emb = np.zeros(dim)
        else:
            emb = word2emb[word]
        word_embs.append(emb) 

    word_embs = np.array(word_embs)

    return word_embs

