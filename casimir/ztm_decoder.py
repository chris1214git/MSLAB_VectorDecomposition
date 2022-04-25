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
            break
        except:
            print('[Error] CUDA Memory Insufficient, retry after 15 secondes.')
            time.sleep(15)

    # Pre-Define Document to check
    # Notice: only for vocabulary size = 8000
    if config['vocabulary_size'] == 8000:
        if config['dataset'] == 'agnews':
            doc_idx = [3041, 23382, 15246, 1387, 16978, 13820, 7266, 14358, 3205, 16899, 8518, 7694, 1914, 19332, 1960, 8148, 25266, 19759, 22715, 23271, 25306, 17074, 8995, 7810, 402, 9763, 555, 1059, 22072, 1579, 25119, 747, 8271, 16746, 9975, 15923, 3710, 1008, 16675, 18786, 24655, 12313, 3910, 1668, 5736, 23431, 15640, 10038, 17669, 24392, 21286, 12068, 16406, 7675, 2127, 5507, 22232, 16273, 21307, 25353, 9323, 9262, 3563, 18829, 13410, 4701, 18191, 10318, 1435, 1200, 19589, 10041, 3924, 7198, 20780, 9951, 11237, 14951, 9583, 2994, 20057, 15798, 214, 9229, 8089, 24561, 3015, 25420, 8846, 11102, 5179, 862, 756, 6269, 19421, 24180, 24280, 87, 5598, 16856, 10416, 916, 7481, 18061, 23881, 5161, 5058, 8153, 8742, 21472, 7819, 6108, 17928, 4079, 11008, 2023, 2437, 20051, 11003, 19180, 20735, 20937, 21412, 12672, 9468, 21197, 1549, 19815, 30, 9337, 16695, 8744, 9821, 9433, 18586, 13324, 3959, 22630, 22443, 474, 22284, 21190, 25004, 9082, 85, 6157, 5906, 6863, 4230, 8565, 24471, 13775, 24822, 14067, 22568, 5215, 22576, 21000, 666, 19812, 7893, 7214, 3402, 6862, 13472, 400, 11084, 7934, 17142, 21585, 535, 9436, 20761, 2594, 3438, 18031, 8285, 17012, 14506, 3726, 3130, 22854, 5027, 24403, 24767, 16487, 15513, 10063, 4269, 25334, 22940, 14085, 691, 11355, 1405, 19863, 9582, 3962, 3604, 17592]
        elif config['dataset'] == '20news':
            doc_idx = [959, 682, 1569, 354, 2162, 3178, 268, 982, 2552, 3296, 701, 1429, 1241, 2228, 1234, 1626, 2892, 1640, 916, 3366, 718, 193, 2341, 3377, 2345, 3259, 2351, 1525, 639, 3734, 100, 2057, 2890, 2898, 2504, 2916, 539, 2659, 3598, 2785, 2654, 3046, 195, 227, 2414, 1202, 2916, 3487, 134, 152, 2945, 768, 3530, 2112, 561, 2856, 2669, 1640, 3545, 2184, 306, 111, 402, 2425, 3546, 1734, 531, 1613, 2010, 3705, 1810, 646, 1502, 843, 1071, 1092, 2460, 3749, 1029, 220, 1729, 615, 991, 2714, 2605, 825, 2495, 2998, 3482, 778, 572, 2125, 1667, 1206, 3229, 1150, 3200, 2966, 2746, 2898, 19, 612, 813, 3305, 1148, 2047, 1230, 778, 1642, 2848, 2879, 3215, 3454, 1149, 2774, 494, 679, 1953, 3167, 2916, 3101, 2263, 885, 2906, 1428, 1474, 3011, 2054, 1217, 3577, 613, 58, 337, 3090, 170, 2126, 481, 85, 795, 901, 2759, 1397, 166, 3604, 2626, 2960, 3401, 1212, 2834, 2577, 1522, 2518, 1584, 3217, 1946, 1573, 2758, 2691, 2522, 1158, 3699, 3208, 3457, 1554, 2037, 1905, 2161, 3689, 1447, 761, 2204, 398, 2099, 352, 664, 2194, 3277, 3046, 294, 3161, 2937, 2823, 2356, 3145, 1194, 224, 1074, 1765, 3152, 617, 1337, 2238, 1375, 3362, 2203, 2428, 1548, 2111, 3660, 2429]
        elif config['dataset'] == 'tweet':
            doc_idx = [448, 580, 547, 248, 191, 782, 64, 489, 446, 157, 628, 312, 830, 633, 99, 15, 637, 342, 454, 644, 741, 133, 829, 726, 639, 253, 575, 745, 107, 590, 301, 482, 786, 453, 674, 776, 536, 369, 680, 627, 424, 585, 440, 566, 670, 281, 678, 663, 106, 84, 279, 316, 685, 795, 564, 535, 360, 568, 785, 386, 654, 387, 303, 719, 746, 361, 40, 639, 324, 684, 324, 567, 77, 628, 658, 601, 684, 33, 353, 106, 750, 226, 711, 585, 753, 374, 298, 588, 488, 736, 557, 497, 429, 167, 322, 22, 69, 464, 528, 635, 3, 439, 480, 265, 744, 640, 711, 274, 118, 322, 192, 483, 709, 599, 788, 601, 232, 646, 310, 46, 325, 186, 321, 161, 561, 218, 259, 602, 66, 339, 54, 83, 664, 107, 682, 552, 556, 137, 780, 516, 589, 266, 464, 792, 429, 254, 493, 360, 165, 109, 235, 464, 404, 784, 68, 448, 308, 686, 526, 2, 323, 162, 454, 490, 253, 389, 134, 370, 106, 526, 473, 85, 400, 640, 129, 152, 454, 453, 52, 204, 127, 369, 440, 449, 219, 655, 404, 782, 508, 41, 26, 204, 108, 365, 54, 516, 699, 272, 196, 210]
    elif config['vocabulart=y_size'] == 5000:
        # if config['dataset'] == 'agnews':
        #     doc_idx = []
        # elif config['dataset'] == '20news':
        #     doc_idx = []
        # elif config['dataset'] == 'tweet':
        #     doc_idx = []
        doc_idx = []
        for idx in range(200):
            doc_idx.append(randint(0, len(validation_set)))
    elif config['vocabulart=y_size'] == 2000:
        # if config['dataset'] == 'agnews':
        #     doc_idx = []
        # elif config['dataset'] == '20news':
        #     doc_idx = []
        # elif config['dataset'] == 'tweet':
        #     doc_idx = []
        doc_idx = []
        for idx in range(200):
            doc_idx.append(randint(0, len(validation_set)))
    else:
        doc_idx = []
        for idx in range(200):
            doc_idx.append(randint(0, len(validation_set)))
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
        for word_idx in range(10):
            print(dataset.idx2token[recon_rank_list[idx][word_idx]])
            record.write(str(dataset.idx2token[recon_rank_list[idx][word_idx]])+'\n')
        print('---------------------------------------')
        record.write('---------------------------------------\n')
        print('[Label] Top 10 Words in Document: ')
        record.write('[Label] Top 10 Words in Document: \n')
        for idx in range(10):
            print(dataset.idx2token[target_rank_list[idx][idx]])
            record.write(str(dataset.idx2token[target_rank_list[idx][idx]])+'\n')
        print('---------------------------------------\n')
        record.write('---------------------------------------\n\n')
