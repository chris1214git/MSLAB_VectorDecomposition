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
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse, io

sys.path.append("../")
from utils.toolbox import get_preprocess_document

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unsupervised keyword retrieve.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--preprocess_config_dir', type=str, default="parameters_baseline")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--cpu_num', type=str, default="1")
    parser.add_argument('--no_keyword', action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    config = vars(args)
    dataset = config['dataset']
    preprocess_config_dir = config['preprocess_config_dir']
    
    with open(os.path.join(f'../chris/{preprocess_config_dir}', f'preprocess_config_{dataset}.json'), 'r') as f:
        preprocess_config = json.load(f)

    # constrain cpu/not working on keybert
    torch.set_num_threads(int(config["cpu_num"]))
        
    # read dataset
    print("loading dataset: {}".format(preprocess_config["dataset"]))
    print("preprocess config", preprocess_config)
    unpreprocessed_docs ,preprocessed_docs = get_preprocess_document(**preprocess_config)
    
    # save folder
    save_folder1 = "precompute_keyword"
    save_folder2 = preprocess_config_dir
    ngram = config['ngram']
    save_folder3 = f"{dataset}_ngram_{ngram}"

    save_folder = os.path.join(save_folder1, save_folder2, save_folder3)
    os.makedirs(save_folder, exist_ok=True)    
    
    with open(os.path.join(f'{save_folder}', f'preprocess_config.json'), 'w') as f:
        json.dump(preprocess_config, f)

    # BOW TFIDF
    vectorizer = TfidfVectorizer(ngram_range=(1, config['ngram']))
    tf_idf_vector = vectorizer.fit_transform(preprocessed_docs)
    bow_vector = tf_idf_vector.copy()
    bow_vector[bow_vector > 0] = 1
    bow_vector[bow_vector < 0] = 0
    vocabulary = np.array(vectorizer.get_feature_names())
    print('vocabulary size', len(vocabulary))
    
    # str to index dict, sync keyphrase result
    stoi = vectorizer.vocabulary_
    np.save(os.path.join(save_folder, "vocabulary.npy"), vocabulary)
    sparse.save_npz(os.path.join(save_folder, "BOW.npz"), bow_vector)
    sparse.save_npz(os.path.join(save_folder, "TFIDF.npz"), tf_idf_vector)
    print("saving vocabulary.npy")       
    print("saving BOW.npz")
    print("saving TFIDF.npz")
    
    if config['no_keyword']:
        exit()
        
    # KeyBERT
    print("KeyBERT extraction ... ...")
    from keybert import KeyBERT
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    keybert_vector = lil_matrix((tf_idf_vector.shape), dtype='float')
    
    for i, doc in enumerate(tqdm(preprocessed_docs)):
        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, config['ngram']), stop_words='english',top_n=10000)
        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            keybert_vector[i, stoi[k[0]]] = k[1]           
    print("saving KeyBERT.npz")
    sparse.save_npz(os.path.join(save_folder, "KeyBERT.npz"), keybert_vector.tocsr())

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

    yake_vector = lil_matrix((tf_idf_vector.shape), dtype='float')

    for i, doc in enumerate(tqdm(preprocessed_docs)):
        keywords = kw_extractor.extract_keywords(doc)

        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            # smaller score, more important. 1 - x for all scores
            yake_vector[i, stoi[k[0]]] = 1 - k[1] 
        # yake_vector[i] = 1 - yake_vector[i] 

    print("saving YAKE.npz")
    sparse.save_npz(os.path.join(save_folder, "YAKE.npz"), yake_vector.tocsr())
