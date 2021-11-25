import torch
import torch.nn as nn

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

def listNet_origin(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor

    Note: ignore zero value by setting padded_value_indicator = 0
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

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

def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
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
