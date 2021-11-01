import matplotlib
from model import Decoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.utils.data import random_split
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append("../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.data_processing import get_process_data

matplotlib.use('Agg')


class DocDataset(Dataset):
    def __init__(self, doc_embedding, doc_tfidf):
        self.doc_tfidf_s = []
        self.vocab_size = len(doc_tfidf[0])
        self.doc_embedding = torch.FloatTensor(doc_embedding)
        for idx in range(len(doc_tfidf)):
            self.doc_tfidf_s.append(
                sorted(range(self.vocab_size), key=lambda k: doc_tfidf[idx][k], reverse=True))
            self.doc_tfidf_s[idx][50] = -1
            # nonzero = 0
            # for value in doc_tfidf[idx]:
            #     if (value > 0):
            #         nonzero += 1
            # self.doc_tfidf_s[idx][nonzero + 1] = -1
        self.doc_tfidf_s = torch.tensor(self.doc_tfidf_s)

    def __getitem__(self, idx):
        return self.doc_embedding[idx], self.doc_tfidf_s[idx]

    def __len__(self):
        return len(self.doc_embedding)

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def training(doc_embedding, doc_tfidf):
    lr = 0.001
    epochs = 100
    batch_size = 64
    vocab_size = len(doc_tfidf[0])
    embedding_dim = len(doc_embedding[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Vocab size:{}, embedding_dim:{}".format(vocab_size, embedding_dim))

    # Preparing training and validation data.
    train_size_ratio = 0.8
    train_size = int(len(doc_embedding) * train_size_ratio)
    test_size = len(doc_embedding) - train_size

    dataset = DocDataset(doc_embedding, doc_tfidf)

    train_dataset, test_dataset = random_split(dataset, lengths=[train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = Decoder(vocab_size=vocab_size,
                    embedding_dim=embedding_dim).to(device)

    loss_function = nn.MultiLabelMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_train_loss, log_val_loss = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(device), target.to(device)
            decoded = model(data)

            loss = loss_function(decoded, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        log_train_loss.append(train_loss / len(train_loader))
        print('[{}/{}] Train Loss:'.format(epoch+1, epochs),
              train_loss / len(train_loader))

        # Evaluate
        model.eval()
        val_loss = 0
        for batch, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # print(data)
            data, target = data.to(device), target.to(device)
            decoded = model(data)

            loss = loss_function(decoded, target)

            val_loss += loss.item()

        log_val_loss.append(val_loss / len(test_loader))
        print('[{}/{}] Validation Loss:'.format(epoch +
              1, epochs), val_loss / len(test_loader))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       "LSTM_{}d_{}-epoch_decoder.pth".format(embedding_dim, epoch+1))

    plt.plot(log_train_loss)
    plt.plot(log_val_loss)
    plt.savefig("loss.png")


def main():
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--embedding_type', type=str, default="LSTM")
    parser.add_argument('--min_word_freq_threshold', type=int, default=5)
    parser.add_argument('--topk_word_freq_threshold', type=int, default=100)

    args = parser.parse_args()
    config = vars(args)

    same_seeds(config["seed"])

    data_dict = get_process_data(config["dataset"], embedding_type=config["embedding_type"],
                                 embedding_dim=config["embedding_dim"])

    doc_embedding, doc_tfidf = data_dict["document_embedding"], data_dict["document_tfidf"]

    print(np.array(doc_embedding).shape)
    print(np.array(doc_tfidf).shape)

    training(doc_embedding, doc_tfidf)


if __name__ == '__main__':
    main()
