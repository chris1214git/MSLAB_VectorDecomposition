import os
import sys
from collections import defaultdict
import math
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.metrics import ndcg_score

def evaluate_sklearn(pred, ans):
    results = {}

    one_hot_ans = np.arange(ans.shape[0])[ans > 0]

    sorted_score = np.argsort(pred)
    for topk in [10, 30, 50]:
        one_hot_pred = sorted_score[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        recall = len(hit) / len(one_hot_ans)
        
        results['F1@{}'.format(topk)] = 2 * percision * recall / (percision + recall) if len(hit) != 0 else 0
        
    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    results['ndcg@10'] = (ndcg_score(ans, pred, k=10))
    results['ndcg@30'] = (ndcg_score(ans, pred, k=30))
    results['ndcg@50'] = (ndcg_score(ans, pred, k=50))
    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))
    
    return results


results = []
train_ratio = 0.8
n_component = sys.argv[1]

document_vectors = np.load("document_vectors.npy", allow_pickle=True)
document_vectors = document_vectors[int(train_ratio * len(document_vectors)):]
reconstruct_vectors = np.load("reconstruct_{}.npy".format(n_component), allow_pickle=True)

print("Document size:{}".format(document_vectors.shape))
print("Reconstruct size:{}".format(reconstruct_vectors.shape))

for doc_id, (original_v, reconstruct_v) in enumerate(tqdm(zip(document_vectors, reconstruct_vectors))):

    res = evaluate_sklearn(reconstruct_v, original_v)
    results.append(res)

results = pd.DataFrame(results)

print("PCA n = {} reconstruct result".format(n_component))

print(results.mean())

print("===============================================")