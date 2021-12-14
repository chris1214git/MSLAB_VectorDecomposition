import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

sys.path.append("../")

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

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, dropout_rate,):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        
        # embedded = pack_padded_sequence(embedded, length, batch_first=True,
        #                                        enforce_sorted=False)
        out, (hidden, cell) = self.lstm(embedded)
        batch_size, seq_size, hidden_size = out.shape

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(batch_size * seq_size, hidden_size)

        # apply dropout
        out = self.fc(self.dropout(out))
        out_feat = out.shape[-1]
        out = out.view(batch_size, seq_size, out_feat)
        return out

    def get_docvec(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = pack_padded_sequence(embedded, length.cpu(), batch_first=True,
                                               enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        doc_vec = torch.sum(out, dim = 1) / (length.reshape(-1,1))
        return doc_vec

def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index=0):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid 

def sequence_loss(y_pred, y_true, mask_index=0):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)



def eval_downstream(model,device):
    model.eval()
    document_vectors = []
    target = data_dict["LSTM_data"]["target_tensor"].numpy()
    for word, length in tqdm(full_loader):
        word = word.to(device)
        length = length.to(device)
        with torch.no_grad():
            vec = model.get_docvec(word, length)
        document_vectors.append(vec)
    document_vectors = torch.cat(document_vectors,dim=0).detach().cpu().numpy()
    
    # LR 
    train_x = document_vectors[train_dataset.indices,:]
    train_y = target[train_dataset.indices]

    test_x = document_vectors[valid_dataset.indices,:]
    test_y = target[valid_dataset.indices]
    
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(train_x, train_y)
    preds = classifier.predict(test_x)
    valid_acc = np.mean(preds == test_y)
    
    return valid_acc


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
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    config = vars(args)

    # load document
    # print(f"Setting seeds:{config['seed']}")
    same_seeds(config["seed"])
    seed_everything(config["seed"])
    data_dict = get_process_data(
        config["dataset"], embedding_dim=config["dim"],
         max_seq_length=config["max_seq_length"],
         embedding_type= "LSTM",
         )

    # prepare pytorch input
    # tokenize_data = sorted(tokenize_data, key = lambda x: len(x), reverse = True)
    seq_length = data_dict["LSTM_data"]["seq_length"]
    paded_context = data_dict["LSTM_data"]["paded_context"]

    # dataset
    dataset = TensorDataset(paded_context, seq_length)
    train_length = int(len(dataset)*0.8)
    valid_length = len(dataset) - train_length

    full_loader = DataLoader(dataset, batch_size=128)
    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_length, valid_length],
        generator=torch.Generator().manual_seed(42)
        )
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(get_freer_gpu())
    vocab_size = len(data_dict["document_word_weight"][0])

    model = LSTM(vocab_size, config["dim"], vocab_size, 0.5).to(device)
    # model.load_state_dict(torch.load("RNN-LM.pt"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    best_loss = 9999999

    for epoch in range(500):
        accuracy = []
        running_loss = []
        model.train()
        for word, length in tqdm(train_loader):
            word = word.to(device)
            length = length.to(device)
            logits = model(word, length)
            logits = logits[:,:-1,:].reshape(-1, vocab_size)
            target = word[:,1:].reshape(-1)
            loss = sequence_loss(logits, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            acc_t = compute_accuracy(logits, target)
            accuracy.append(acc_t)
            running_loss.append(loss.item())
        
        avg_loss = np.mean(running_loss)
        print(f"[Epoch {epoch:02d}] Train Accuracy:{np.mean(accuracy):.4f} Train Loss:{avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                "RNN-LM.pt"
            )
        if (epoch+1) % 20 ==0:
            valid_acc = eval_downstream(model,device)
            print(f"Validation accuracy:{valid_acc:.4f}")
