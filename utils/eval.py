import torch

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