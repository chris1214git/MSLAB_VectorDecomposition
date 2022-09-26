from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import torch.nn as nn
import torch
import os
import sys
import argparse

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.data_processing import get_process_data

# fix random seed

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, dropout_rate,):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = pack_padded_sequence(embedded, length, batch_first=True,
                                               enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction

    def get_docvec(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = pack_padded_sequence(embedded, length, batch_first=True,
                                               enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        return hidden[-1]


def evaluate(model, test_loader, device):
    accuracy = []
    for word, length, target in test_loader:
        word, target = word.to(device), target.to(device)
        with torch.no_grad():
            logits = model(word, length)
        hits = logits.argmax(dim=1).eq(target)
        accuracy.append(hits)
    return torch.cat(accuracy).float().mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_word_freq_threshold', type=int, default=5)
    parser.add_argument('--topk_word_freq_threshold', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--word2embedding_path', type=str,
                        default="glove.6B.100d.txt")

    args = parser.parse_args()
    config = vars(args)

    # load document
    # print(f"Setting seeds:{config['seed']}")
    same_seeds(config["seed"])
    data_dict = get_process_data(
        config["dataset"], max_seq_length=config["max_seq_length"])

    # prepare pytorch input
    # tokenize_data = sorted(tokenize_data, key = lambda x: len(x), reverse = True)
    seq_length = data_dict["LSTM_data"]["seq_length"]
    paded_context = data_dict["LSTM_data"]["paded_context"]
    target_tensor = data_dict["LSTM_data"]["target_tensor"]

    # dataset
    dataset = TensorDataset(paded_context, seq_length, target_tensor)
    train_length = int(len(dataset)*0.8)
    valid_length = len(dataset) - train_length

    full_loader = DataLoader(dataset, batch_size=128)
    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_length, valid_length])
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = len(data_dict["document_tfidf"][0])
    num_classes = data_dict["dataset"]["num_classes"]

    model = LSTM(vocab_size, config["dim"], num_classes, 0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(100):
        accuracy = []
        model.train()
        for word, length, target in tqdm(train_loader):
            word, target = word.to(device), target.to(device)

            logits = model(word, length)
            loss = loss_function(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            hits = logits.argmax(dim=1).eq(target)
            accuracy.append(hits)

        accuracy = torch.cat(accuracy).float().mean()
        print(f"[Epoch {epoch:02d}] Train Accuracy:{accuracy:.4f}")
        valid_acc = evaluate(model, valid_loader, device)
        print(f"[Epoch {epoch:02d}] Valid Accuracy:{valid_acc:.4f}")

    # save document embedding
    model.eval()
    document_representation = []
    for word, length, _ in tqdm(full_loader):
        word = word.to(device)
        with torch.no_grad():
            vectors = model.get_docvec(word, length)
            vectors = vectors.detach().cpu().numpy().tolist()
        document_representation.extend(vectors)

    print("Saving document vectors")
    np.save(
        f"docvec_20news_LSTM_{config['dim']}d.npy", document_representation)
