import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 2048),
            nn.Tanh(),
            nn.Linear(2048, vocab_size),
        )

    def forward(self, embeddings):
        return self.decoder(embeddings)
