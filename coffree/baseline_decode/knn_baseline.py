import re
import sys
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

sys.path.append("../..")

from utils.data_processing import get_process_data
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, get_freer_gpu, show_settings, split_data

def evaluate_Decoder(config, model, X_test, y_test):
    results = defaultdict(list)
        
    # predict all data
    pred = torch.tensor(model.predict(X_test))
    y = torch.tensor(y_test)

    # Precision
    precision_scores = retrieval_precision_all(pred, y, k=config["topk"])
    for k, v in precision_scores.items():
        results['precision@{}'.format(k)].append(v)
    
    # NDCG
    ndcg_scores = retrieval_normalized_dcg_all(pred, y, k=config["topk"])
    for k, v in ndcg_scores.items():
        results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results

def main():
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="IMDB")
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
    args = parser.parse_args()
    config = vars(args)

    show_settings(config)

    same_seeds(config["seed"])

    data_dict = get_process_data(config["dataset"])

    importance_score = data_dict["document_word_weight"]
    docvec = np.load("../../data/docvec_IMDB_SBERT_768d.npy")

    print("TFIDF dim:{}".format(importance_score.shape))
    print("Doc vector shape:{}".format(docvec.shape))

    X_train, X_test, y_train, y_test = train_test_split(docvec, importance_score, test_size=0.2, random_state=config["seed"])
    
    model = KNeighborsRegressor(n_neighbors=config["n_neighbors"])

    model.fit(X_train, y_train)

    res = evaluate_Decoder(config, model, X_test, y_test)

    for key,val in res.items():
        print(f"{key}:{val:.4f}")


if __name__ == '__main__':
    main()