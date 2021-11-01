import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import ndcg_score
from model import Decoder

sys.path.append("../..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.data_processing import get_process_data

class DocDataset(Dataset):
    def __init__(self, doc_embedding, doc_tfidf):
        self.doc_embedding = torch.FloatTensor(doc_embedding)
        self.doc_tfidf = torch.FloatTensor(doc_tfidf)
        assert len(doc_embedding) == len(doc_tfidf)

    def __getitem__(self, idx):
        return self.doc_embedding[idx], self.doc_tfidf[idx]

    def __len__(self):
        return len(self.doc_embedding)


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


def main():
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--epoch', type=str)
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--embedding_type', type=str, default="LSTM")
    parser.add_argument('--min_word_freq_threshold', type=int, default=5)
    parser.add_argument('--topk_word_freq_threshold', type=int, default=100)
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])

    args = parser.parse_args()
    config = vars(args)

    data_dict = get_process_data(
        config["dataset"], embedding_type=config["embedding_type"], embedding_dim=config["embedding_dim"])

    doc_embedding, doc_tfidf = data_dict["document_embedding"], data_dict["document_tfidf"]

    vocab_size = len(doc_tfidf[0])
    embedding_size = len(doc_embedding[0])

    # Preparing training and validation data.
    train_size_ratio = 0.8
    train_size = int(len(doc_embedding) * train_size_ratio)
    test_size = len(doc_embedding) - train_size

    dataset = DocDataset(doc_embedding, doc_tfidf)

    train_dataset, test_dataset = random_split(dataset, lengths=[train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # evaluate
    print("Start evaluatiing")
    model = Decoder(vocab_size, embedding_size).to(device)
    model_path = "LSTM_{}d_{}-epoch_decoder.pth".format(
        config["embedding_dim"], config["epoch"])
    print("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    results = []
    with torch.no_grad():
        for batch, (data, target) in enumerate(tqdm(test_loader)):
            decoded = model(data.to(device)).cpu()
            for idx in range(len(decoded)):
                if sum(decoded[idx]) != 0:
                    res = evaluate_sklearn(decoded[idx], target[idx], config)
                    results.append(res)

    results_m = pd.DataFrame(results).mean()
    print('------' + "LSTM_{}d_{}-epoch_decoder.pth".format(
        config["embedding_dim"], config["epoch"]) + '------')
    print(results_m)
    print('-------------------------------')


if __name__ == '__main__':
    main()
