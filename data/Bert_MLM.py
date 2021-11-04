import os
import sys
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from transformers import BertTokenizer, BertForMaskedLM
from sklearn.linear_model import LogisticRegression
from utils.data_loader import load_document

class BertDataset(Dataset):

    def __init__(self, documents, targets) -> None:
        super().__init__()
        self.documents = documents
        self.targets = torch.LongTensor(targets)
        assert len(documents) == len(targets)

    def __len__(self):
        return len(documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.targets[idx]


class BertForMaskLM(nn.Module):
    def __init__(self, device, target_num: int = 20):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained(
            'bert-base-uncased').to(device)

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

    def forward(self, documents):

        inputs = self.generate_mask_input(documents).to(self.device)

        outputs = self.model(**inputs)

        return outputs

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


def train_model(device, train_loader, target_num):
    iterations = 100
    model = BertForMaskLM(device, target_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iters in range(iterations):
        # Training a epoch.
        model.train()
        train_loss = 0
        for idx, (data, targets) in enumerate(tqdm(train_loader, desc="Training")):
            outputs = model(data)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print("Epoch:{}/{}, training loss:{}".format(iters +
              1, iterations, train_loss / len(train_loader)))

    return model


def generate_document_vector(model, data_loader):
    model.eval()
    document_vectors = []

    with torch.no_grad():
        for documents, targets in tqdm(data_loader):
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

    args = parser.parse_args()
    config = vars(args)

    same_seeds(config["seed"])

    # Prepare dataset.
    dataset = load_document(config["dataset"])
    documents, targets, target_num = dataset["documents"], dataset["target"], dataset["num_classes"]

    train_ratio = 0.8
    train_length = int(len(documents) * train_ratio)
    test_length = len(documents) - train_length

    dataset = BertDataset(documents, targets)
    full_loader = DataLoader(dataset,  batch_size=60,
                             shuffle=False, pin_memory=True)

    train_data, test_data = random_split(dataset, lengths=[train_length, test_length],
                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_data, batch_size=60, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=60,
                             shuffle=False, pin_memory=True)

    # Start training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(device, train_loader, target_num)

    document_vectors = generate_document_vector(model, full_loader)
    print("Saving document vectors")
    np.save(f"docvec_20news_BertLM.npy", document_vectors)

    valid_acc = evaluate_downstream(document_vectors, targets, train_data, test_data)
    print(f"Validation accuracy:{valid_acc:.4f}")

