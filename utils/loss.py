import torch
import torch.nn as nn
import torch.nn.functional as F

def ListNet(y_pred, y_true, eps=1e-10):
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded. 
    #     ex: same as above.
    # (3) eps: a small number to avoid error when computing log operation. 
    #     ex: log0 will cause error while log(0+eps) will not.

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    # pred_smax = torch.nn.functional.softmax(y_pred, dim=1)
    # true_smax = torch.nn.functional.softmax(y_true, dim=1)

    pred = y_pred + eps
    pred_log = torch.log(pred)

    return torch.mean(torch.sum(-y_true * pred_log, dim=1))

def topk_ListNet(y_pred, y_true, y_rank, eps=1e-10, topk=64):
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded .
    #     ex: same as above.
    # (3) y_rank: the index of the desceding ranking of y_true.
    #     ex: y_true = [40, 20, 30, 10], then y_rank should be [0, 2, 1, 3].
    # (4) eps: a small number to avoid error when computing log operation. 
    #     ex: log0 will cause error while log(0+eps) will not.
    # (5) topk: the top k elements which we want to compute.
    #     ex: in 20 news, 64 is the better since the max length of document is also 64.
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    y_rank = y_rank.clone()

    y_pred = torch.gather(y_pred, 1, y_rank[:, :topk])
    y_true = torch.gather(y_true, 1, y_rank[:, :topk])
  
  
    preds_smax = y_pred + eps
    preds_log = torch.log(preds_smax)
    return torch.mean(torch.sum(-y_true * preds_log, dim=1))

def RankLoss(y_pred, y_true):
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded .
    #     ex: same as above.
    # warning: this loss function may cause memory limit exceeded when the size of input vector become larger.
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    return torch.relu(-(y_pred.view(-1,1)*y_pred.view(1,-1)) * (y_true.view(-1,1)*y_true.view(1,-1)))

def MSERankLoss(y_pred, y_true):
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded .
    #     ex: same as above.
    pred_rank, rank_p = torch.sort(y_pred, dim=1, descending=True)
    label_rank, rank_l = torch.sort(y_true, dim=1, descending=True)

    return torch.nn.MSELoss().forward(torch.FloatTensor(rank_p), torch.FloatTensor(rank_l))

def ListNet_origin(y_pred, y_true, eps=1e-10):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

def ListMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    
    Note: ignore zero value by setting padded_value_indicator = 0
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def MultiLabelMarginLossCustom(y_pred, y_true_rank, fixed_topk=50, alpha=1):
    """
    MultiLabelMarginLoss add positive pairs
    y_pos_id -> index of positive target, the same as MultiLabelMarginLoss before -1
    y_neg_id -> index of negative target
    alpha -> magnitude of positive pairs compared to negative pairs, since normally negative pairs are 100 times more than positive pairs
    """
    device = y_pred.device    
    y_pos_id = y_true_rank[:, :fixed_topk]
    y_neg_id = y_true_rank[:, fixed_topk:]
    
    y_pos = y_pred.gather(1, y_pos_id)
    y_neg = y_pred.gather(1, y_neg_id)

    m = y_pos.view(y_pos.shape[0], y_pos.shape[1], 1) - y_neg.view(y_neg.shape[0], 1, y_neg.shape[1])
    m = 1 - m
    l = torch.max(m, torch.zeros(m.shape).to(device))
    l = torch.sum(l)
    
    mp = y_pos.view(y_pos.shape[0], y_pos.shape[1], 1) - y_pos.view(y_pos.shape[0], 1, y_pos.shape[1])

    mp = 1 - mp - torch.eye(mp.shape[-1]).to(device)
    lp = torch.max(mp, torch.zeros(mp.shape).to(device))
    lp = torch.sum(lp)
    
    loss = (l + alpha * lp) / y_pred.shape[1] / y_pred.shape[0]
    
    return loss

def MultiLabelMarginLossCustomV(y_pred, y_true_rank, y_true_topk, alpha=1):
    """
    variable positive candidates version
    
    MultiLabelMarginLoss add positive pairs
    y_pos_id -> index of positive target, the same as MultiLabelMarginLoss before -1
    y_neg_id -> index of negative target
    alpha -> magnitude of positive pairs compared to negative pairs, since normally negative pairs are 100 times more than positive pairs
    """    
    loss = 0
    device = y_pred.device
    
    y_pos_id = []
    y_neg_id = []
    
    for i in range(len(y_pred)):
        topk = min(y_true_topk[i], 100)
        y_pos_id.append(y_true_rank[i, :topk])
        y_neg_id.append(y_true_rank[i, topk:])
    
    for i in (range(y_pred.shape[0])):
        pred, pos_id, neg_id = y_pred[i], y_pos_id[i], y_neg_id[i]
        
        m = pred[pos_id].view(-1, 1) - pred[neg_id].view(1, -1)
        m = 1 - m
        l = torch.max(m, torch.zeros(m.shape).to(device))
        l = torch.sum(l)
        
        mp = pred[pos_id].view(-1, 1) - pred[pos_id].view(1, -1)
        mp = 1 - mp - torch.eye(len(pos_id)).to(device)
        lp = torch.max(mp, torch.zeros(mp.shape).to(device))
        lp = torch.sum(lp)
        loss += (l + alpha * lp) / y_pred.shape[1]
        
    loss /= y_pred.shape[0]
    
    return loss

def MSE(y_pred, y_true):
    loss = nn.MSELoss(reduction='none')(y_pred, y_true)
    return torch.mean(torch.sum(loss, dim=1))

def ListNet2(y_pred, y_true):
    y_pred_min = torch.min(y_pred, dim=1)[0].view(-1, 1)
    y_true_min = torch.min(y_true, dim=1)[0].view(-1, 1)
    
    y_pred = y_pred + y_pred_min + 10
    y_true = y_true + y_true_min + 10

    pred_log = torch.log(y_pred)

    return torch.mean(torch.sum(-y_true * pred_log, dim=1))
