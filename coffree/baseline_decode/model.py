import torch
import torch.nn as nn


class Decoder_only(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 4096),
            nn.Tanh(),
            nn.Linear(4096, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Decoder_wordembed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.word_embedding = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )

    def init_weights(self):
        print("Initializing decoder weights...")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        docvec = self.transform(x)
        decoded = torch.sigmoid(torch.matmul(docvec, self.word_embedding))
        return decoded

    def load_pretrianed(self, word_embedding):
        self.word_embedding = nn.Parameter(torch.FloatTensor(word_embedding))
