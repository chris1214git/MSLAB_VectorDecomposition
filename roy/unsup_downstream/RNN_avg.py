import argparse
import sys
import os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

sys.path.append("../")

from utils.data_processing import get_process_data
from utils.Loss import ListNet

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
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_dim, 1024),
            # nn.Tanh(),
            # nn.Linear(1024, 4096),
            # nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        
        embedded = pack_padded_sequence(embedded, length.cpu(), batch_first=True,
                                               enforce_sorted=False)
        packed_output, (hidden, ct) = self.lstm(embedded)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        batch_size, seq_size, hidden_size = out.shape
        
        # for TF-IDF decoder
        doc_vec = torch.sum(out, dim = 1) / (length.reshape(-1,1))
        decoded_output = self.decoder(doc_vec)

        # for LM output
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(batch_size * seq_size, hidden_size)

        # apply dropout
        out = self.fc(self.dropout(out))
        out_feat = out.shape[-1]
        out = out.view(batch_size, seq_size, out_feat)
        
        return decoded_output, out

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
    for word, length,_ in tqdm(full_loader):
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


def evaluate_sklearn(pred, ans, config):
    results = {}

    one_hot_ans = np.arange(ans.shape[0])[ans > 0]
    sorted_prediction = np.argsort(pred)
    for topk in config["topk"]:
        one_hot_pred = sorted_prediction[-topk:]
        hit = np.intersect1d(one_hot_pred, one_hot_ans)
        percision = len(hit) / topk
        # print(percision)
        recall = len(hit) / len(one_hot_ans)
        # print(recall)
        f1 = 2 * percision * recall / \
            (percision + recall) if (percision + recall) > 0 else 0

        results['F1@{}'.format(topk)] = f1

    ans = ans.reshape(1, -1)
    pred = pred.reshape(1, -1)
    for topk in config["topk"]:
        results['ndcg@{}'.format(topk)] = ndcg_score(ans, pred, k=topk)

    results['ndcg@all'] = (ndcg_score(ans, pred, k=None))

    return results

def evaluate_decompose(model, device, config):
    model.eval()
    result = []
    for word, length, label in tqdm(valid_loader):
        word = word.to(device)
        length = length.to(device)
        with torch.no_grad():
            vec = model.get_docvec(word, length)
            decoded = model.decoder(vec)
        for idx in range(len(label)):
            res = evaluate_sklearn(decoded.cpu()[idx], label.cpu()[idx],config)
            result.append(res)
    result = pd.DataFrame(result).mean()
    return result

class BertDataset(Dataset):

    def __init__(self, paded_context, seq_length, labels) -> None:
        super().__init__()
        self.paded_context = paded_context
        self.seq_length = seq_length
        self.labels = torch.tensor(labels)
        assert len(paded_context) == len(seq_length) and len(paded_context) == len(labels)

    def __len__(self):
        return len(self.paded_context)

    def __getitem__(self, idx):
        return self.paded_context[idx], self.seq_length[idx], self.labels[idx]

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
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])

    args = parser.parse_args([])
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
    raw_data = data_dict["dataset"]
    documents, targets, target_num = raw_data["documents"], raw_data["target"], raw_data["num_classes"]
    labels = data_dict["document_word_weight"]

    # dataset
    dataset = BertDataset(paded_context, seq_length, labels)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    best_loss = 9999999

    for epoch in range(500):
        running_loss = []
        running_lmloss = []
        running_listnetloss = []
        listnet_weight = 1 if epoch > 0 else 0
        model.train()
        for batch in tqdm(train_loader):
            batch = [i.to(device) for i in batch]
            word, length, label = batch
            decoded_output, logits = model(word, length)
            label = torch.nn.functional.normalize(label.to(device), dim=1)
            decoded = torch.nn.functional.normalize(decoded_output, dim=1)
            listnet_loss = ListNet(decoded, label)
            logits = logits[:,:-1,:].reshape(-1, vocab_size)
            target = word[:,1:].reshape(-1)
            lm_loss = sequence_loss(logits, target)
            loss = listnet_weight * listnet_loss + lm_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss.append(loss.item())
            running_listnetloss.append(listnet_loss.item())
            running_lmloss.append(lm_loss.item())
        
        avg_loss = np.mean(running_loss)
        avgLM_loss = np.mean(running_lmloss)
        avgListnet_loss = np.mean(running_listnetloss)
        print(f"[Epoch {epoch:02d}]  Train Loss:{avg_loss:.4f} LM-Loss:{avgLM_loss:.4f} Listnet-Loss:{avgListnet_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                "RNN-TFIDF.pt"
            )
        if (epoch+1) % 20 ==0:
            valid_acc = eval_downstream(model,device)
            decompose_res = evaluate_decompose(model, device, config)
            print(f"Validation accuracy:{valid_acc:.4f}")
            print("Decompose result:")
            print(decompose_res)
