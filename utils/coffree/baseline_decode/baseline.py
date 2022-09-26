import re
import sys
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from collections import defaultdict

sys.path.append("../..")

from model import Decoder_only, Decoder_wordembed, Decoder_GVAE
from utils.loss import ListNet
from utils.data_processing import get_process_data
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all
from utils.toolbox import same_seeds, get_freer_gpu, show_settings, split_data

class IDEDataset(Dataset):
    def __init__(self, 
                 doc_vectors,
                 doc_tfidf):
        
        assert len(doc_vectors) == len(doc_tfidf)

        self.doc_vectors = torch.FloatTensor(doc_vectors)
        self.doc_tfidf = torch.FloatTensor(doc_tfidf)
        
    def __getitem__(self, idx):
        return self.doc_vectors[idx], self.doc_tfidf[idx]

    def __len__(self):
        return len(self.doc_vectors)

def evaluate_Decoder(config, model, data_loader, device):
    results = defaultdict(list)
    model.eval()
        
    # predict all data
    for data in data_loader:
        doc_embs, target = data
        
        doc_embs = doc_embs.to(device)
        target = target.to(device)
                
        pred = model(doc_embs)
    
        # Precision
        precision_scores = retrieval_precision_all(pred, target, k=config["topk"])
        for k, v in precision_scores.items():
            results['precision@{}'.format(k)].append(v)
        
        # NDCG
        ndcg_scores = retrieval_normalized_dcg_all(pred, target, k=config["topk"])
        for k, v in ndcg_scores.items():
            results['ndcg@{}'.format(k)].append(v)
        
    for k in results:
        results[k] = np.mean(results[k])

    return results

def train_model(decoder, device, config, train_loader, valid_loader):
    # early stop settings 
    stop_rounds = 3
    no_improvement = 0
    best_score = None
    optimizer = torch.optim.Adam(decoder.parameters(), lr = config["lr"])

    for epoch in range(config["epochs"]):
        # Training
        decoder.train()
        train_loss = []
        for idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = [i.to(device) for i in batch]
            doc_embs, target = batch
            target = torch.nn.functional.normalize(target.to(device), dim=1)
            decoded = torch.nn.functional.normalize(decoder(doc_embs), dim=1)
            loss = ListNet(decoded, target)
            if (config["decoder"] == "WGVAE" and idx == len(train_loader)-1):
                loss += decoder.train_reconstruct(device) * config["penalty"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

        print(f"[Epoch {epoch+1:02d}]")
        res = evaluate_Decoder(config, decoder, valid_loader, device)
        for key,val in res.items():
            print(f"{key}:{val:.4f}")
            
        # early stopping 
        current_score = res["precision@10"]
        if best_score == None:
            best_score = current_score
            continue
        if current_score < best_score:
            no_improvement += 1
        if no_improvement >= stop_rounds:
            print("Early stopping...")
            break 
        if current_score > best_score:
            no_improvement = 0
            best_score = current_score

    return decoder


def main():
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--dataset', type=str, default="IMDB")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default=get_freer_gpu())
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--penalty', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--decoder', type=str, default="Only")  # (1) Only: directly docode (2) wordembed: use wording embedding information (3) WGVAE: wordembed with graph VAE
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
    args = parser.parse_args()
    config = vars(args)

    show_settings(config)

    same_seeds(config["seed"])

    device = config["gpu"]

    data_dict = get_process_data(config["dataset"])

    importance_score = data_dict["document_word_weight"]
    docvec = np.load("../../data/docvec_IMDB_SBERT_768d.npy")

    print("TFIDF dim:{}".format(importance_score.shape))
    print("Doc vector shape:{}".format(docvec.shape))

    dim = 768
    vocab_size = importance_score.shape[1]
    print("Vocab size:{}".format(vocab_size))

    dataset = IDEDataset(docvec, importance_score)

    train_loader, valid_loader, test_loader = split_data(dataset, config)

    if config["decoder"] == 'wordembed':
        decoder = Decoder_wordembed(input_dim=dim, hidden_dim=config["hidden_dim"], output_dim=vocab_size).to(device)
        decoder.init_weights()
    elif config["decoder"] == "WGVAE":
        raw_documents=data_dict["dataset"]["documents"]
        decoder = Decoder_GVAE(documents=raw_documents, input_dim=dim, hidden_dim=config["hidden_dim"], output_dim=vocab_size).to(device)
        decoder.init_weights()
    else:
        decoder = Decoder_only(input_dim=dim, output_dim=vocab_size).to(device)

    decoder = train_model(decoder, device, config, train_loader, valid_loader)

    print("Testing...")
    res = evaluate_Decoder(config, decoder, test_loader, device)
    for key,val in res.items():
        print(f"{key}:{val:.4f}")


if __name__ == '__main__':
    main()