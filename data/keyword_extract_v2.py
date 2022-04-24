# ## Extract keywords
# 1. KeyBert
# 2. YAKE
# 3. pke
#     https://github.com/boudinfl/pke
import sys
import os
import argparse
import numpy as np
import json
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

sys.path.append("../")
from utils.toolbox import get_preprocess_document

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unsupervised keyword retrieve.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--cpu_num', type=str, default="1")
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    config = vars(args)
    dataset_name = config['dataset']
    with open(os.path.join('../chris/parameters', f'preprocess_config_{dataset_name}.json'), 'r') as f:
        preprocess_config = json.load(f)

    # constrain cpu/not working on keybert
    torch.set_num_threads(int(config["cpu_num"]))
        
    # read dataset
    print("loading dataset: {}".format(preprocess_config["dataset_name"]))
    print("preprocess config", preprocess_config)
    unpreprocessed_docs ,preprocessed_docs = get_preprocess_document(**preprocess_config)
    preprocess_config['ngram'] = config['ngram']
    
    # save folder
    save_folder1 = "precompute_keyword"
    save_folder2 = "keyword"
    for k, v in preprocess_config.items():
        save_folder2 = save_folder2 + f"_{k}_{v}"
    save_folder = os.path.join(save_folder1, save_folder2)
    os.makedirs(save_folder, exist_ok=True)    
    
    # BOW TFIDF
    vectorizer = TfidfVectorizer(ngram_range=(1, config['ngram']))
    tf_idf_vector = vectorizer.fit_transform(preprocessed_docs).todense()
    bow_vector = tf_idf_vector.copy()
    bow_vector[bow_vector > 0] = 1
    bow_vector[bow_vector < 0] = 0
    vocabulary = np.array(vectorizer.get_feature_names())

    # str to index dict, sync keyphrase result
    stoi = vectorizer.vocabulary_
    np.save(os.path.join(save_folder, "vocabulary.npy"), vocabulary)
    np.save(os.path.join(save_folder, "BOW.npy"), bow_vector)
    np.save(os.path.join(save_folder, "TFIDF.npy"), tf_idf_vector)
    print('vocabulary size', len(vocabulary))
    
    # KeyBERT
    print("KeyBERT extraction ... ...")
    from keybert import KeyBERT
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    keybert_vector = np.zeros(tf_idf_vector.shape)
    
    for i, doc in enumerate(tqdm(preprocessed_docs)):
        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, config['ngram']), stop_words='english',top_n=10000)
        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            keybert_vector[i, stoi[k[0]]] = k[1]           
    print("saving KeyBERT.npy")
    np.save(os.path.join(save_folder, "KeyBERT.npy"), keybert_vector)

    # YAKE
    # all score are negative!
    print("YAKE extraction ... ...")
    import yake
    language = "en"
    max_ngram_size = config['ngram']
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 10000

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, \
                                         dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,\
                                         features=None)

    yake_vector = np.zeros(tf_idf_vector.shape)

    for i, doc in enumerate(tqdm(preprocessed_docs)):
        keywords = kw_extractor.extract_keywords(doc)

        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            yake_vector[i, stoi[k[0]]] = k[1] 
        # smaller score, more important. x-1 for all scores
        yake_vector[i] = -yake_vector[i] 
    print("saving YAKE.npy")
    np.save(os.path.join(save_folder, "YAKE.npy"), yake_vector)
    
    
    