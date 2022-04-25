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
    parser.add_argument('--dataset_name', type=str, default="20news")
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--mxa_df', type=float, default=1.0)
    parser.add_argument('--vocabulary_size', type=int, default=1000)
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
    parser.add_argument('--check_document', type=bool, default=True)
    parser.add_argument('--check_auto', type=bool, default=True)
    parser.add_argument('--check_nums', type=int, default=500)
    args = parser.parse_args()
    
    config = vars(args)
    config["dataset_name"] = config["dataset"]
    show_settings(config)
    record_settings(config)
    same_seeds(config["seed"])

    # data preprocessing
    unpreprocessed_corpus ,preprocessed_corpus = get_preprocess_document(**config)
    texts = [text.split() for text in preprocessed_corpus]

    # generating document embedding
    doc_embs, doc_model = get_preprocess_document_embs(preprocessed_corpus, config['encoder'])

    # Decode target & Vocabulary
    labels, vocabularys= get_preprocess_document_labels(preprocessed_corpus)
    id2token = {k: v for k, v in zip(range(0, len(vocabularys[config['target']])), vocabularys[config['target']])}

    # prepare dataset
    tp = TopicModelDataPreparation(contextualized_model=doc_embs, target=config['target'])
    dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_corpus, text_for_doc2vec=texts, decode_target=labels[config['target']], vocab=vocabularys[config['target']], id2token=id2token)
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
                model = CombinedTM(bow_size=len(tp.vocab), contextual_size=contextual_size, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            elif config['model'] == 'mlp':
                model = MLPDecoder(bow_size=len(tp.vocab), contextual_size=contextual_size, num_epochs=config['epochs'], config=config, texts=texts,vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            else:
                model = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=contextual_size, n_components=config['topic_num'], num_epochs=config['epochs'], config=config, texts=texts, vocab = tp.vocab, word_embeddings=word_embeddings, idx2token=dataset.idx2token)
            model.fit(training_set, validation_set)
        except:
            print('[Error] Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)

    # Pre-Define Document to check
    if config['dataset'] == 'agnews':
        print('[Error]')
    elif config['dataset'] == '20news':
        print('[Error]')
    elif config['dataset'] == 'tweet':
        doc_idx = [654, 194, 352, 178, 251, 385, 804, 782, 834, 627, 114, 345, 203, 592, 131, 534, 340, 716, 531, 70, 71, 117, 373, 543, 469, 409, 777, 486, 614, 38, 729, 736, 455, 840, 591, 106, 72, 468, 713, 173, 682, 199, 767, 103, 308, 477, 793, 468, 645, 673, 484, 733, 262, 339, 368, 110, 754, 254, 140, 232, 617, 344, 14, 375, 649, 134, 732, 298, 320, 134, 576, 32, 349, 576, 312, 310, 725, 510, 139, 731, 75, 821, 471, 762, 707, 755, 773, 219, 475, 277, 716, 66, 611, 280, 735, 829, 17, 28, 423, 341, 438, 235, 828, 54, 76, 392, 290, 705, 518, 448, 144, 355, 14, 459, 95, 264, 703, 274, 363, 391, 488, 446, 324, 91, 178, 238, 68, 70, 525, 323, 169, 79, 49, 29, 25, 722, 393, 746, 709, 806, 335, 308, 562, 447, 227, 710, 301, 291, 411, 846, 631, 564, 457, 358, 470, 26, 203, 225, 135, 75, 750, 818, 450, 332, 9, 249, 256, 847, 420, 353, 528, 518, 808, 565, 557, 619, 56, 719, 815, 558, 162, 527, 408, 301, 767, 134, 95, 109, 619, 580, 320, 483, 205, 324, 153, 261, 348, 78, 372, 12]
    # visualize documents
    for idx in doc_idx:
        # get recontruct result
        recon_list, target_list, doc_list = model.get_reconstruct(validation_set)

        # get ranking index
        recon_rank_list = np.zeros((len(recon_list), len(tp.vocab)), dtype='float32')
        target_rank_list = np.zeros((len(recon_list), len(tp.vocab)), dtype='float32')
        for i in range(len(recon_list)):
            recon_rank_list[i] = np.argsort(recon_list[i])[::-1]
            target_rank_list[i] = np.argsort(target_list[i])[::-1]

        # show info
        record = open('./'+config['dataset']+'_'+config['model']+'_'+config['encoder']+'_'+config['target']+'_document.txt', 'a')
        doc_topics_distribution = model.get_doc_topic_distribution(validation_set)
        doc_topics = model.get_topic_lists()[np.argmax(doc_topics_distribution[idx])]
        print('Documents ', idx)
        record.write('Documents '+str(idx)+'\n')
        print(doc_list[idx])
        record.write(doc_list[idx])
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
            print(dataset.idx2token[recon_rank_list[idx][idx]])
            record.write(str(dataset.idx2token[recon_rank_list[idx][idx]])+'\n')
        print('---------------------------------------')
        record.write('---------------------------------------\n')
        print('[Label] Top 10 Words in Document: ')
        record.write('[Label] Top 10 Words in Document: \n')
        for idx in range(10):
            print(dataset.idx2token[target_rank_list[idx][idx]])
            record.write(str(dataset.idx2token[target_rank_list[idx][idx]])+'\n')
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
