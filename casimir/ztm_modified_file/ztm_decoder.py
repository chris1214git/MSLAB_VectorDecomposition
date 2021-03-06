import os
import sys
import nltk
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, random_split
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, calculate_word_embeddings_tensor, load_word2emb
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
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--check_document', type=bool, default=False)
    parser.add_argument('--check_auto', type=bool, default=True)
    parser.add_argument('--check_nums', type=int, default=50)
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

    # word embedding preparation
    word_embeddings = calculate_word_embeddings_tensor(load_word2emb("./data/glove.6B.200d.txt"), tp)

    # build model & training
    if config['model'] == 'CombinedTM':
        model = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab=vocab, tp_vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
    else:
        model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab=vocab, tp_vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
    model.fit(training_set, validation_set)

    # visualize documents
    check_nums = config['check_nums']
    while config['check_document']:
        # get recontruct result
        recon_list, doc_list = model.get_reconstruct(validation_set)

        # get ranking index
        recon_rank_list = np.zeros((len(recon_list), len(tp.vocab)), dtype='float32')
        for i in range(len(recon_list)):
            recon_rank_list[i] = np.argsort(recon_list[i])[::-1]

        # show info
        record = open('./'+config['model']+'_'+config['target']+'_document.txt', 'a')
        doc_idx = random.randint(0, len(recon_list))
        doc_topics_distribution = model.get_doc_topic_distribution(validation_set)
        doc_topics = model.get_topic_lists()[np.argmax(doc_topics_distribution[doc_idx])]
        print('Documents ', doc_idx)
        record.write('Documents '+str(doc_idx)+'\n')
        print(doc_list[doc_idx])
        record.write(doc_list[doc_idx])
        print('---------------------------------------')
        record.write('\n---------------------------------------\n')
        print('Topic of Document: ')
        record.write('Topic of Document: \n')
        print(doc_topics)
        record.write(str(doc_topics))
        print('---------------------------------------')
        record.write('---------------------------------------\n')
        print('Top 10 Words in Document: ')
        record.write('Top 10 Words in Document: \n')
        for idx in range(10):
            print(dataset.idx2token[recon_rank_list[doc_idx][idx]])
            record.write(str(dataset.idx2token[recon_rank_list[doc_idx][idx]])+'\n')
        print('---------------------------------------\n')
        record.write('---------------------------------------\n\n')
        print('Press any key to continue / exit [e]')
        check_nums -= 1

        # determine done or not
        if config['check_auto'] and (check_nums <= 0):
            break
        elif config['check_auto'] and (check_nums > 0):
            continue
        else:
            check = input()
            if check == 'e':
                break
