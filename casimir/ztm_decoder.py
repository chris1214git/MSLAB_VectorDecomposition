import sys
import os
import torch
import nltk
import argparse
from torch.utils.data import DataLoader, random_split
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from utils.data_loader import load_document
from utils.toolbox import same_seeds, show_settings

sys.path.append("../")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="ZTM")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--target', type=str, default='BoW')
    parser.add_argument('--topic_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--topk', type=int, nargs='+', default=[10, 30, 50])
    args = parser.parse_args()
    
    config = vars(args)
    show_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    raw_documents = load_document(config['dataset'])["documents"]
    sp = WhiteSpacePreprocessing(raw_documents, stopwords_language='english')
    preprocessed_documents, unpreprocessed_corpus, vocab, _ = sp.preprocess()
    texts = [text.split() for text in preprocessed_documents]

    # prepare dataset
    tp = TopicModelDataPreparation(contextualized_model="paraphrase-distilroberta-base-v1", target=config['target'])
    dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
    training_length = int(len(dataset) * config['ratio'])
    validation_length = len(dataset) - training_length
    training_set, validation_set = random_split(dataset, lengths=[training_length, validation_length],generator=torch.Generator().manual_seed(42))

    # build model & training
    if config['model'] == 'CombinedTM':
        model = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab=vocab, idx2token=dataset.idx2token)
    else:
        model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab=vocab, idx2token=dataset.idx2token)
    model.fit(training_set, validation_set)
