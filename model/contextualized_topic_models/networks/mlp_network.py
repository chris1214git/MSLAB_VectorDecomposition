import torch
from torch import nn
from torch.nn import functional as F

class mlp_network(nn.Module):

    ### casimir
    # (1) Add parameter vocab_size
    def __init__(self, bert_size, vocab_size=0):
        super(mlp_network, self).__init__()
        self.vocab_size = vocab_size
       
        self.network = nn.Sequential(
            nn.Linear(bert_size, bert_size*4),
            nn.BatchNorm1d(bert_size*4),
            nn.Sigmoid(),
            nn.Linear(bert_size*4, vocab_size),
            nn.BatchNorm1d(vocab_size),
            nn.Sigmoid(),
        )

    def forward(self, x_bert):
        recon_dist = self.network(x_bert)

        return recon_dist


