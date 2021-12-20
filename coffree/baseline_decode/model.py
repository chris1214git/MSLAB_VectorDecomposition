import re
from math import log
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE, VGAE
from sklearn.feature_extraction.text import TfidfVectorizer


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

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Decoder_GVAE(nn.Module):
    def __init__(self, documents, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=10, stop_words="english")
        vectorizer.fit(documents)
        self.word2idx = vectorizer.vocabulary_
        self.index2word = {}
        for i in self.word2idx:
            self.index2word[self.word2idx[i]] = i
        self.vocab = set(list(self.word2idx.keys()))
        self.doc_list = [self.document_filter(doc) for doc in tqdm(documents, desc="Delete word from raw document...")]
        
        self.word_embedding = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.train_data = self.build_graph()
        self.graph_autoencoder = VGAE(VariationalGCNEncoder(hidden_dim, 16))

    def document_filter(self, doc_raw):
        PATTERN = r"(?u)\b\w\w+\b"
        doc = re.findall(PATTERN, doc_raw.lower())
        return [x for x in doc if x in self.vocab]

    def train_reconstruct(self, device):
        self.train_data.to(device)
        z = self.graph_autoencoder.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.graph_autoencoder.recon_loss(z, self.train_data.edge_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.graph_autoencoder.kl_loss()
        return loss

    def build_graph(self):
        '''
        Here are some features needed to build graph input for GVAE.
        (1) word_embedding [hidden_dim, vocab_size]
        (2) word_index dict
        (3) edge_index
        '''
        window_size = 5
        weight = []
        windows, row, col = [], [], []

        for words in tqdm(self.doc_list, desc="Building windows..."):
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)

        word_freq = {}
        word_pair_count = {}
        for window in tqdm(windows, desc="Building graph matrix..."):
            appeared = set()
            for i in range(len(window)):
                if window[i] not in appeared:
                    if window[i] in word_freq:
                        word_freq[window[i]] += 1
                    else:
                        word_freq[window[i]] = 1
                    appeared.add(window[i])
                if i != 0:
                    for j in range(0, i):
                        word_i = window[i]
                        word_i_id = self.word2idx[word_i]
                        word_j = window[j]
                        word_j_id = self.word2idx[word_j]
                        if word_i_id == word_j_id:
                            continue
                        word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                        if word_pair_str in word_pair_count:
                            word_pair_count[word_pair_str] += 1
                        else:
                            word_pair_count[word_pair_str] = 1
                        word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                        if word_pair_str in word_pair_count:
                            word_pair_count[word_pair_str] += 1
                        else:
                            word_pair_count[word_pair_str] = 1

        num_window = len(windows)

        for key in tqdm(word_pair_count, desc='Constructing Edge...'):
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_freq[self.index2word[i]]
            word_freq_j = word_freq[self.index2word[j]]
            pmi = log((1.0 * count / num_window) /
                    (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            if count >= 10:
                row += [i, j]
                col += [j, i]

            weight.append(pmi)

        edge_index = torch.tensor([row, col], dtype=torch.long)

        print('# of Node: {}\n# of Edge: {}'.format(len(self.word2idx), edge_index.size(1)))

        return Data(x=self.word_embedding, edge_index=edge_index)

    def init_weights(self):
        print("Initializing decoder weights...")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        docvec = self.transform(x)
        decoded = torch.sigmoid(torch.matmul(docvec, torch.t(self.word_embedding)))
        return decoded
