import numpy as np
import torch
from collections import defaultdict
from sklearn.linear_model import LogisticRegression


def _dcg(target: torch.tensor):
    """Computes Discounted Cumulative Gain for input tensor."""
    denom = torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
    return (target / denom).sum(dim=-1)

def retrieval_normalized_dcg_all(preds: torch.tensor, target: torch.tensor, k=None):
    """Computes `Normalized Discounted Cumulative Gain`_ (for information retrieval).
    Args:
    (1) preds: tensor with 2d shape
    (2) target: tensor with 2d shape
    (3) k: a list of integer, automated padding pred.shapes[1]
    Return:
    (1) ndcg_scores: dict
        key -> k, value -> average ndcg score
    """
    k = [preds.shape[-1]] if k is None else k + [preds.shape[-1]]
    
    assert preds.shape == target.shape and max(k) <= preds.shape[-1]
    
    if not isinstance(k, list):
        raise ValueError("`k` has to be a list of positive integer or None")

    sorted_target = target.gather(1, torch.argsort(preds, dim=-1, descending=True))
    ideal_target = torch.sort(target, descending=True)[0]
    
    ndcg_scores = {}
    for topk in k:
        sorted_target_k = sorted_target[:,:topk]
        ideal_target_k = ideal_target[:,:topk]
        
        ideal_dcg_k = _dcg(ideal_target_k)
        target_dcg_k = _dcg(sorted_target_k)
        
        # filter undefined scores
        target_dcg_k /= ideal_dcg_k
        
        if topk == preds.shape[-1]:
            topk = 'all'
        ndcg_scores[topk] = target_dcg_k.mean().item()
        
    return ndcg_scores

def retrieval_precision_all(preds, target, k = [10]):
    """Computes `TopK precision`_ (for information retrieval).
    Note:
        select topk pred
        consider all positive target as ground truth
    Args:
    (1) preds: tensor with 2d shape
    (2) target: tensor with 2d shape
    (3) k: a list of integer
    Return:
    (1) precision_scores: dict
        key -> k, value -> average precision score
    """
    assert preds.shape == target.shape and max(k) <= preds.shape[-1]
    
    if not isinstance(k, list):
        raise ValueError("`k` has to be a list of positive integer")
        
    precision_scores = {}
    target_onehot = target > 0
    
    for topk in k:
        relevant = target_onehot.gather(1, preds.topk(topk, dim=-1)[1])
        relevant = relevant.sum(axis=1).float()
        relevant /= topk    
        precision_scores[topk] = relevant.mean().item()
    
    return precision_scores

def semantic_precision_all(preds, target, word_embeddings, tp_vocab, k = [10], th = 0.7, display_word_result=False):
    """Computes `TopK precision`_ (for information retrieval).
    Note:
        select topk pred
        consider all positive target as ground truth
        one ground truth only count once

    Args:
    (1) preds: tensor with 2d shape
    (2) target: tensor with 2d shape
    (3) word_embeddings: word_embeddings matrix tensor with 2d shape(v, d)
    (4) k: a list of integer
    (5) th: threshold of word embeddings cosine similarity
    (5) display_word_result: whether to display ground truth, prediction & hit words

    Return:
    (1) precision_scores: dict
        key -> k, value -> average precision score

    """
    assert preds.shape == target.shape and max(k) <= preds.shape[-1]
    # assert preds.shape[1] == word_embeddings.shape[0]
    
    if not isinstance(k, list):
        print(k)
        print(word_embeddings.shape)
        raise ValueError("`k` has to be a list of positive integer")
        
    precision_scores = {}
    target_onehot = target > 0
    vocab = np.array(tp_vocab)
    word_result = defaultdict(list)
    
    for topk in k:
        relevants = []
        for i in range(preds.shape[0]):
            preds_word_emb = word_embeddings[preds[i].topk(topk)[1]]
            target_word_emb = word_embeddings[target_onehot[i]]
            
            similarity_matrix = torch.zeros(preds_word_emb.shape[0], target_word_emb.shape[0])
            for j in range(preds_word_emb.shape[0]):
                similarity_matrix[j] = torch.nn.functional.cosine_similarity(torch.from_numpy(preds_word_emb[j]).view(1, -1), torch.from_numpy(target_word_emb))
                
            max_similarity_score, max_similarity_idx = torch.max(similarity_matrix, dim=1)
            max_similarity_idx = max_similarity_idx[max_similarity_score >= th]
            relevant = len(torch.unique(max_similarity_idx)) / topk
            relevants.append(relevant)
            if topk == 10 and display_word_result:
                word_result['ground truth'].append(vocab[target_onehot[i].cpu().numpy()])
                word_result['prediction'].append(vocab[preds[i].topk(topk)[1].cpu().numpy()])
                word_result['hit'].append(vocab[target_onehot[i].cpu().numpy()][max_similarity_idx.cpu().numpy()])
                
        precision_scores[topk] = np.mean(relevants)
    
    return precision_scores, word_result

def evaluate_downstream(document_vectors, targets, train_dataset, test_dataset):
    # LR 
    train_x = document_vectors[train_dataset.indices,:]
    train_y = targets[train_dataset.indices]

    test_x = document_vectors[test_dataset.indices,:]
    test_y = targets[test_dataset.indices]
    
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(train_x, train_y)
    preds = classifier.predict(test_x)
    valid_acc = np.mean(preds == test_y)
    
    return valid_acc

def precision_recall_f1_all(preds, target):
    """Computes precision, recall, f1 for binary target    
    Args:
    (1) preds: tensor with 2d shape, raw output
    (2) target: tensor with 2d shape, multi-hot vector
    Return:
    (1) ndcg_scores: dict
        key -> k, value -> average ndcg score
    """
    preds = torch.sigmoid(preds)
    # binarize prediction
    pred_b = (preds >= 0.5).float()
    target_b = target
    hit_num = torch.sum((pred_b == 1) & (target_b == 1), axis=1)
    gt_num = torch.sum((target_b == 1), axis=1)
    pred_num = torch.sum((pred_b == 1), axis=1)

    precision = hit_num / pred_num
    recall = hit_num / gt_num
    f1 = 2 * precision * recall / (precision + recall)
    precision = torch.nan_to_num(precision, nan=0)
    recall = torch.nan_to_num(recall, nan=0)
    f1 = torch.nan_to_num(f1, nan=0)

    precision = torch.mean(precision).cpu().item()
    recall = torch.mean(recall).cpu().item()
    f1 = torch.mean(f1).cpu().item()

    return precision, recall, f1 