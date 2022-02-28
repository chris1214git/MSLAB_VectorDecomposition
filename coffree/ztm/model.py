import torch
import torch.nn as nn
import torch.nn.functional as F
from contextualized_topic_models.models.ctm import ZeroShotTM

import sys
sys.path.append("../..")
from utils.loss import ListNet

class ZTM(nn.Module):
    
    def __init__(self, bow_size, contextual_size, n_components, hidden_sizes, activation):
        super().__init__()
        self._ZeroShotTM = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size, 
                    n_components=n_components,hidden_sizes=hidden_sizes,activation=activation)
        self.decodernet = self._ZeroShotTM.model

    def forward(self, target, doc_embs, is_train=False):
        prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists,_ = self.decodernet(target, doc_embs)

        if (is_train):
            self.kl_loss, self.rl_loss = self._ZeroShotTM._loss(target, word_dists, prior_mean, prior_variance,
                            posterior_mean, posterior_variance, posterior_log_variance)
        return word_dists
        
    def calculate_loss(self, pred, target):
        topic_loss = (self.kl_loss * 1e-3 + self.rl_loss).sum()
        return topic_loss

class ZTM_topic_embed(nn.Module):
    
    def __init__(self, bow_size, contextual_size, n_components, hidden_sizes, activation):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(bow_size)
        self.word_embedding = nn.Parameter(torch.randn(128 + 768, bow_size))
        self.topic_embedding = nn.Parameter(torch.randn(n_components, 128))
        self._ZeroShotTM = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size, 
                    n_components=n_components,hidden_sizes=hidden_sizes,activation=activation)
        self.decodernet = self._ZeroShotTM.model

    def forward(self, target, doc_embs, is_train=False):
        prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists,_ = self.decodernet(target, doc_embs)

        if (is_train):
            self.kl_loss, self.rl_loss = self._ZeroShotTM._loss(target, word_dists, prior_mean, prior_variance,
                            posterior_mean, posterior_variance, posterior_log_variance)

        # Topic probability: batch_size x n_components
        topic_prob = F.softmax(
            self.decodernet.reparameterize(posterior_mean, posterior_log_variance), dim=1).detach()
        
        # batch_size x 128
        topic_info = torch.sigmoid(torch.matmul(topic_prob, self.topic_embedding))

        # embeded: batch_size x (128 + 768)
        embeded = torch.cat((topic_info, doc_embs), dim=1)

        word_dists = torch.sigmoid(self.batch_norm(torch.matmul(embeded, self.word_embedding)))
        
        return word_dists
        
    def calculate_loss(self, pred, target):
        pred = torch.nn.functional.normalize(pred, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)
        listnet_loss = ListNet(pred, target)
        topic_loss = (self.kl_loss * 1e-3 + self.rl_loss).sum()
        return listnet_loss + topic_loss


class ZTM_decoder(nn.Module):
    
    def __init__(self, bow_size, contextual_size, n_components, hidden_sizes, activation):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bow_size, 4096),
            nn.BatchNorm1d(4096),
            nn.Sigmoid(),
            nn.Linear(4096, bow_size),
            nn.BatchNorm1d(bow_size),
            nn.Sigmoid(),
        )
        self._ZeroShotTM = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size, 
                    n_components=n_components,hidden_sizes=hidden_sizes,activation=activation)
        self.decodernet = self._ZeroShotTM.model

    def forward(self, target, doc_embs, is_train=False):
        prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists,_ = self.decodernet(target, doc_embs)

        if (is_train):
            self.kl_loss, self.rl_loss = self._ZeroShotTM._loss(target, word_dists, prior_mean, prior_variance,
                            posterior_mean, posterior_variance, posterior_log_variance)
        word_dists = self.decoder(word_dists)
        return word_dists
        
    def calculate_loss(self, pred, target):
        pred = torch.nn.functional.normalize(pred, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)
        listnet_loss = ListNet(pred, target)
        topic_loss = (self.kl_loss * 1e-3 + self.rl_loss).sum()
        return topic_loss + listnet_loss

class ZTM_word_embed(nn.Module):
    
    def __init__(self, bow_size, contextual_size, n_components, hidden_sizes, activation):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(bow_size)
        self.word_embedding = nn.Parameter(torch.randn(4096 + 768, bow_size))
        self.transform = nn.Sequential(
            nn.Linear(bow_size, 4096),
            nn.BatchNorm1d(4096),
            nn.Sigmoid(),
        )
        self._ZeroShotTM = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size, 
                    n_components=n_components,hidden_sizes=hidden_sizes,activation=activation)
        self.decodernet = self._ZeroShotTM.model

    def forward(self, target, doc_embs, is_train=False):
        prior_mean, prior_variance, posterior_mean, posterior_variance,\
            posterior_log_variance, word_dists,_ = self.decodernet(target, doc_embs)

        if (is_train):
            self.kl_loss, self.rl_loss = self._ZeroShotTM._loss(target, word_dists, prior_mean, prior_variance,
                            posterior_mean, posterior_variance, posterior_log_variance)

        docvec = self.transform(word_dists)
        embeded = torch.cat((docvec, doc_embs), dim=1)
        word_dists = torch.sigmoid(self.batch_norm((torch.matmul(embeded, self.word_embedding))))
        return word_dists
        
    def calculate_loss(self, pred, target):
        pred = torch.nn.functional.normalize(pred, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)
        listnet_loss = ListNet(pred, target)
        topic_loss = (self.kl_loss * 1e-3 + self.rl_loss).sum()
        return topic_loss + listnet_loss