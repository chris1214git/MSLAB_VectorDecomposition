import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

sys.path.append("../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import ndcg_score

from transformers import BertTokenizer, BertForMaskedLM
from sklearn.linear_model import LogisticRegression
from utils.Loss import ListNet
from utils.data_processing import get_process_data

class BertDataset(Dataset):

    def __init__(self, documents, targets, labels) -> None:
        super().__init__()
        self.documents = documents
        self.targets = torch.LongTensor(targets)
        self.labels = torch.tensor(labels)
        assert len(documents) == len(targets) and len(documents) == len(labels)

    def __len__(self):
        return len(documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.targets[idx], self.labels[idx]


class Bert(nn.Module):
    def __init__(self, device, output_dim):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained(
            'bert-base-uncased').to(device)
        self.decoder = nn.Sequential(
            nn.Linear(768, 1024),
            # nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(1024, 4096),
            # nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(4096, output_dim),
            # nn.Dropout(p=0.5),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, documents):
        embeddings = self.get_docvec(documents)
        return self.decoder(embeddings)

    def generate_mask_input(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', 
                                max_length=128, truncation=True, padding=True)
        inputs['labels'] = inputs.input_ids.detach().clone()
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        
        return inputs

    def train_decoder(self, documents, labels):
        labels = torch.nn.functional.normalize(labels.to(self.device), dim=1)

        embeddings = self.get_docvec(documents)

        outputs = torch.nn.functional.normalize(self.decoder(embeddings), dim=1)

        decode_loss = ListNet(outputs, labels)
        
        return decode_loss

    def train_maskLM(self, documents):

        inputs = self.generate_mask_input(documents).to(self.device)

        outputs = self.model(**inputs)

        return outputs.loss

    def get_docvec(self, documents):
        inputs = self.tokenizer(documents, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to(self.device)
        embedding = self.model.bert(**inputs).last_hidden_state[:, 0, :]
        return embedding


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_sklearn(pred, ans, config):
    results = {}

    one_hot_ans = np.arange(ans.shape[0])[ans > 0]

    for topk in config["topk"]:
        one_hot_pred = np.argsort(pred)[-topk:]
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


def train_model(config, device, train_loader, output_dim):
    iterations = 20
    model = Bert(device, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iters in range(iterations):
        # Training a epoch.
        model.train()
        train_loss = 0
        for idx, (data, targets, labels) in enumerate(tqdm(train_loader, desc="Training")):
            decode_loss = model.train_decoder(data, labels)
            if (config["mlm_pretrain"] == "True"):
                mlm_loss = model.train_maskLM(data)
                loss = decode_loss + mlm_loss
            else:
                loss = decode_loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print("Epoch:{}/{}, training loss:{}".format(iters +
              1, iterations, train_loss / len(train_loader)))

    return model


def generate_document_vector(model, data_loader):
    model.eval()
    document_vectors = []

    with torch.no_grad():
        for documents, targets, _ in tqdm(data_loader):
            vec = model.get_docvec(documents)
            document_vectors.append(vec)

    document_vectors = torch.cat(
        document_vectors, dim=0).detach().cpu().numpy()
    return document_vectors


def evaluate_downstream(document_vectors, targets, train_dataset, test_dataset):
    # LR 
    train_x = document_vectors[train_dataset.indices,:]
    train_y = targets[train_dataset.indices]

    test_x = document_vectors[test_dataset.indices,:]
    test_y = targets[test_dataset.indices]
    
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(train_x, train_y)
    preds = classifier.predict(test_x)
    valid_acc = np.mean(preds == test_y)
    
    return valid_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--label', type=str, default="tfidf")
    parser.add_argument('--mlm_pretrain', type=str, default="True")
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])

    args = parser.parse_args()
    config = vars(args)

    same_seeds(config["seed"])

    # Prepare dataset.
    data_dict = get_process_data(config["dataset"])
    dataset = data_dict["dataset"]
    documents, targets, target_num = dataset["documents"], dataset["target"], dataset["num_classes"]

    if (config["label"] == "attention_score"):
        labels = np.load("document_20news_attention_weight.npy")
    elif (config["label"] == "tfidf"):
        labels = data_dict["document_tfidf"]
    else:
        labels = None
    print("Using label {}".format(config["label"]))

    train_ratio = 0.8
    train_length = int(len(documents) * train_ratio)
    test_length = len(documents) - train_length

    dataset = BertDataset(documents, targets, labels)
    full_loader = DataLoader(dataset,  batch_size=60,
                             shuffle=False, pin_memory=True)

    train_data, test_data = random_split(dataset, lengths=[train_length, test_length],
                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_data, batch_size=60, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=60,
                             shuffle=False, pin_memory=True)

    # Start training.
    output_dim = labels.shape[-1]
    print("output dim:{}".format(output_dim))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(config, device, train_loader, output_dim)

    document_vectors = generate_document_vector(model, full_loader)
    print("Saving document vectors")
    np.save(f"docvec_20news_Bert_doced.npy", document_vectors)

    valid_acc = evaluate_downstream(document_vectors, targets, train_data, test_data)
    print(f"Validation accuracy:{valid_acc:.4f}")


    model.eval()
    results = []
    with torch.no_grad():
        for batch, (data, _, target) in enumerate(tqdm(test_loader)):
            decoded = model(data).cpu()
            for idx in range(len(decoded)):
                if sum(decoded[idx]) != 0:
                    res = evaluate_sklearn(decoded[idx], target[idx], config)
                    results.append(res)

        results_m = pd.DataFrame(results).mean()
        print(results_m)
        print('-------------------------------')

