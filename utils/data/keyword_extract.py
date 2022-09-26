# ## Extract keywords
# 1. KeyBert
# 2. YAKE
# 3. pke
#     https://github.com/boudinfl/pke
import sys
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

sys.path.append("../")
from utils.data_processing import get_process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='unsupervised keyword retrieve.')
    parser.add_argument('--dataset', type=str, default="IMDB")
    parser.add_argument('--cpu_num', type=str, default="1")
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    config = vars(args)
    
    # save folder
    save_folder = "precompute_keyword"
    os.makedirs(save_folder, exist_ok=True)
    
    # constrain cpu/not working on keybert
    torch.set_num_threads(int(config["cpu_num"]))
    
    # read dataset
    print("loading dataset: {}".format(config["dataset"]))
    data_dict = get_process_data(config["dataset"])
    doc_raw = data_dict['dataset']['documents']
    
    # TFIDF
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=10, stop_words="english")
    doc_tfidf = vectorizer.fit_transform(doc_raw).todense()
    # str to index dict, sync keyphrase result
    stoi = vectorizer.vocabulary_
    np.save(os.path.join(save_folder, "keyword_{}_TFIDF".format(config["dataset"])), doc_tfidf)

    # KeyBERT
    print("KeyBERT extraction ... ...")
    from keybert import KeyBERT
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    doc_bertkeyword = np.zeros(doc_tfidf.shape)
    
    for i, doc in enumerate(tqdm(doc_raw)):
        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english',top_n=10000)
        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            doc_bertkeyword[i, stoi[k[0]]] = k[1]           
    print("saving keyword_{}_KeyBERT.npy".format(config["dataset"]))
    np.save(os.path.join(save_folder, "keyword_{}_KeyBERT".format(config["dataset"])), doc_bertkeyword)

    # YAKE
    # all score are negative!
    print("YAKE extraction ... ...")
    import yake
    language = "en"
    max_ngram_size = 1
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 10000

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,                                            dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)

    doc_yakekeyword = np.zeros(doc_tfidf.shape)

    for i, doc in enumerate(tqdm(doc_raw)):
        keywords = kw_extractor.extract_keywords(doc)

        for k in keywords:
            # skip words not in vocab
            if k[0] not in stoi:
                continue
            doc_yakekeyword[i, stoi[k[0]]] = k[1] 
            # smaller score, more important. x-1 for all scores
            doc_yakekeyword[i] = -doc_yakekeyword[i] 
    print("saving keyword_{}_YAKE.npy".format(config["dataset"]))
    np.save(os.path.join(save_folder, "keyword_{}_YAKE".format(config["dataset"])), doc_yakekeyword)
    
    
    