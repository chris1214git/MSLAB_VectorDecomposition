import os
import sys
import nltk
import time
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader, random_split

sys.path.append("../")
from model.contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from model.contextualized_topic_models.models.mlp import MLPDecoder
from model.contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation, calculate_word_embeddings_tensor, load_word2emb
from model.contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from utils.data_loader import load_document
from utils.toolbox import same_seeds, show_settings, record_settings, get_preprocess_document, get_preprocess_document_embs, get_preprocess_document_labels, get_word_embs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(8)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='document decomposition.')
    parser.add_argument('--model', type=str, default="ZTM")
    parser.add_argument('--dataset', type=str, default="20news")
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--mxa_df', type=float, default=1.0)
    parser.add_argument('--vocabulary_size', type=int, default=8000)
    parser.add_argument('--min_doc_word', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='roberta')
    parser.add_argument('--target', type=str, default='tf-idf')
    parser.add_argument('--topic_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--check_document', type=bool, default=False)
    parser.add_argument('--check_auto', type=bool, default=True)
    parser.add_argument('--check_nums', type=int, default=500)
    args = parser.parse_args()
    
    config = vars(args)
    show_settings(config)
    record_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # generating document embedding
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    # Decode target & Vocabulary
    labels, vocabularys, id2token = get_preprocess_document_labels(preprocessed_corpus)

    # prepare dataset
    tp = TopicModelDataPreparation(contextualized_model=doc_embs, target=config['target'])
    dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_corpus, decode_target=labels[config['target']], vocab=vocabularys[config['target']], id2token=id2token)
    training_length = int(len(dataset) * config['ratio'])
    validation_length = len(dataset) - training_length
    training_set, validation_set = random_split(dataset, lengths=[training_length, validation_length],generator=torch.Generator().manual_seed(42))

    # word embedding preparation
    word_embeddings = calculate_word_embeddings_tensor(load_word2emb("../data/glove.6B.300d.txt"), tp)

    # Define document embeddings dimension
    if config['encoder'] == 'doc2vec':
        contextual_size = 200
    elif config['encoder'] == 'average':
        contextual_size = 300
    else:
        contextual_size = 768
    
    # build model & training
    while True:
        try:
            if config['model'] == 'CombinedTM':
                model = CombinedTM(bow_size=len(tp.vocab), contextual_size=contextual_size, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, tp_vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            elif config['model'] == 'mlp':
                model = MLPDecoder(bow_size=len(tp.vocab), contextual_size=contextual_size, num_epochs=config['epochs'], config=config, texts=texts,tp_vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            else:
                model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=contextual_size, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, tp_vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            model.fit(training_set, validation_set)
            break
        except:
            print("[Error] Memory not enough, ")
            time.sleep(15)


    # visualize documents
    check_nums = config['check_nums']
    while config['check_document']:
        # get recontruct result
        recon_list, target_list, doc_list = model.get_reconstruct(validation_set)

        # get ranking index
        recon_rank_list = np.zeros((len(recon_list), len(tp.vocab)), dtype='float32')
        target_rank_list = np.zeros((len(recon_list), len(tp.vocab)), dtype='float32')
        for i in range(len(recon_list)):
            recon_rank_list[i] = np.argsort(recon_list[i])[::-1]
            target_rank_list[i] = np.argsort(target_list[i])[::-1]

        # show info
        record = open('./'+config['dataset']+config['model']+'_'+'_'+config['target']+'_document.txt', 'a')
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
        print('[Predict] Top 10 Words in Document: ')
        record.write('[Predict] Top 10 Words in Document: \n')
        for idx in range(10):
            print(dataset.idx2token[recon_rank_list[doc_idx][idx]])
            record.write(str(dataset.idx2token[recon_rank_list[doc_idx][idx]])+'\n')
        print('---------------------------------------')
        record.write('---------------------------------------\n')
        print('[Label] Top 10 Words in Document: ')
        record.write('[Label] Top 10 Words in Document: \n')
        for idx in range(10):
            print(dataset.idx2token[target_rank_list[doc_idx][idx]])
            record.write(str(dataset.idx2token[target_rank_list[doc_idx][idx]])+'\n')
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
