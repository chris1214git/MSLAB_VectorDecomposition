import torch
import torch.nn as nn

def topk_MSELoss(y_pred, y_true, y_rank, topk=64):
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    y_rank = y_rank.clone()

    y_pred = torch.gather(y_pred, 1, y_rank[:, :topk])
    y_true = torch.gather(y_true, 1, y_rank[:, :topk])

    return torch.mean((y_pred - y_true)**2)

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

def MythNet(y_pred, y_true, eps=1e-10):
    # (1) y_pred: the decoded vector. 
    #     ex: tfidf score of each word in certain document.
    # (2) y_true: the vector before encoded. 
    #     ex: same as above.
    # (3) eps: a small number to avoid error when computing log operation. 
    #     ex: log0 will cause error while log(0+eps) will not.

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    #y_pred = torch.sigmoid(y_pred) 
    y_pred = torch.nn.functional.normalize(y_pred, dim=1)
    # y_true = torch.nn.functional.softmax(y_true, dim=1) 
    y_true = torch.nn.functional.normalize(y_true, dim=1)
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
