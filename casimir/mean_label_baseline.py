import sys
import os
import torch
import numpy as np
import nltk
import argparse
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append("../")
from utils.toolbox import get_free_gpu, same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs
from utils.eval import retrieval_normalized_dcg_all, retrieval_precision_all, semantic_precision_all, retrieval_precision_all_v2, semantic_precision_all_v2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(8)

class SimpleDataset(Dataset):
    def __init__(self, predict, target):
        
        assert len(predict) == len(target)
        self.predict = torch.FloatTensor(predict)
        self.target = torch.FloatTensor(target)        
        
    def __getitem__(self, idx):
        return self.predict[idx], self.target[idx]

    def __len__(self):
        return len(self.predict)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="MeanLabel")
    parser.add_argument('--activation', type=str, default="sigmoid")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--use_pos', type=bool, default=True)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--max_df', type=float, default=1.0)
    parser.add_argument('--vocab_size', type=int, default=0)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='bert')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--topic_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--check_document', type=bool, default=False)
    parser.add_argument('--check_auto', type=bool, default=True)
    parser.add_argument('--check_nums', type=int, default=500)
    args = parser.parse_args()
    
    config = vars(args)
    same_seeds(config["seed"])

    # Parameter
    if config['dataset'] == '20news':
        config['min_df'], config['max_df'], config['min_doc_word'] = 50, 1.0, 15
    elif config['dataset'] == 'agnews':
        config['min_df'], config['max_df'], config['min_doc_word'] = 100, 1.0, 15
    elif config['dataset'] == 'IMDB':
        config['min_df'], config['max_df'], config['min_doc_word'] = 100, 1.0, 15
    elif config['dataset'] == 'tweet':
        config['min_df'], config['max_df'], config['min_doc_word'] = 5, 1.0, 15
    
    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
    id2token = {k: v for k, v in zip(range(0, len(vocabularys[config['target']])), vocabularys[config['target']])}
    token2id = {v: k for k, v in id2token.items()}

    # word embedding preparation
    word_embeddings = get_word_embs(vocabularys[config['target']], id2token=id2token, data_type='tensor')

    # predic
    predict = labels[config['target']].mean(axis=0)
    predict = np.tile(predict, (labels[config['target']].shape[0], 1))
    
    # prepare dataset
    dataset = SimpleDataset(predict, labels[config['target']])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True,)

    # calculate score
    device = get_free_gpu()
    results = defaultdict(list)
    for batch, (pred, label) in enumerate(dataloader):
        pred, label = pred.to(device), label.to(device)
        # Semantic Prcision for reconstruct
        precision_scores, word_result = semantic_precision_all(pred, label, word_embeddings, vocabularys[config['target']], k=config['topk'], th = config['threshold'])
        for k, v in precision_scores.items():
            results['[Recon] Semantic Precision v1@{}'.format(k)].append(v)
        precision_scores, word_result = semantic_precision_all_v2(pred, label, word_embeddings, vocabularys[config['target']], k=config['topk'], th = config['threshold'])
        for k, v in precision_scores.items():
            results['[Recon] Semantic Precision v2@{}'.format(k)].append(v)
                        
        # Precision for reconstruct
        precision_scores = retrieval_precision_all(pred, label, k=config['topk'])
        for k, v in precision_scores.items():
            results['[Recon] Precision v1@{}'.format(k)].append(v)
        precision_scores = retrieval_precision_all_v2(pred, label, k=config['topk'])
        for k, v in precision_scores.items():
            results['[Recon] Precision v2@{}'.format(k)].append(v)

        # NDCG for reconstruct
        ndcg_scores = retrieval_normalized_dcg_all(pred, label, k=config['topk'])
        for k, v in ndcg_scores.items():
            results['[Recon] ndcg@{}'.format(k)].append(v)
    
    for k in results:
        results[k] = np.mean(results[k])

    # save result
    record = open('./'+config['dataset']+'_'+config['model']+'_'+config['target']+'.txt', 'a')
    for key,val in results.items():
        print(f"{key}:{val:.4f}")
        record.write(f"{key}:{val:.4f}\n")