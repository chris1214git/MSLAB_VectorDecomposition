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
    elif config['vocabulary_size'] == 5000:
        if config['dataset'] == 'agnews':
            doc_idx = [7016, 15691, 12754, 23327, 20364, 3329, 14927, 20861, 23753, 8652, 18234, 2185, 5381, 2503, 20924, 21042, 21321, 13028, 14437, 19717, 21763, 17883, 2503, 114, 23436, 9803, 12988, 7312, 10851, 25188, 22661, 5268, 9457, 5377, 6990, 8144, 768, 21680, 7969, 13086, 24281, 12703, 18200, 10562, 21415, 19949, 17596, 13612, 17538, 7805, 18717, 7037, 3600, 2433, 11075, 15525, 19250, 6531, 24742, 24709, 6140, 15989, 3717, 17830, 11788, 10988, 23334, 8281, 9786, 5781, 7556, 2455, 15919, 21644, 13101, 2013, 20193, 7962, 1411, 12961, 22734, 21336, 2532, 17263, 4056, 24391, 5244, 11795, 3686, 15168, 4351, 23682, 7147, 3813, 20248, 24697, 12434, 11642, 11913, 22551, 22426, 9634, 3417, 7209, 3721, 11324, 20533, 7918, 12796, 7437, 3725, 19650, 15601, 11022, 9345, 21621, 16253, 25354, 10671, 270, 18826, 8095, 8189, 10407, 23392, 11830, 19326, 20498, 11530, 19093, 2504, 4295, 19167, 12437, 3212, 7029, 2300, 16836, 11763, 18676, 15565, 8628, 5907, 6470, 5495, 1943, 21274, 9004, 905, 3280, 13049, 14056, 20429, 13750, 24887, 5887, 16670, 5345, 3282, 17497, 5781, 19293, 25466, 6208, 17081, 17688, 22909, 24501, 10925, 10867, 24033, 23983, 19416, 11383, 20741, 9266, 1721, 5188, 17271, 16818, 21615, 12473, 24727, 5291, 15551, 16726, 882, 2680, 13441, 10522, 13091, 18597, 15834, 4653, 17780, 21565, 3043, 14426, 15434, 18646]
        elif config['dataset'] == '20news':
            doc_idx = [1001, 3089, 2502, 2236, 3701, 2668, 1612, 2880, 3624, 2223, 1472, 1616, 1553, 379, 3522, 3626, 1417, 578, 543, 2139, 2849, 1420, 3561, 2975, 2640, 1269, 35, 244, 2756, 1606, 2163, 1617, 2728, 3614, 963, 190, 2715, 3312, 2149, 2424, 3322, 845, 1483, 2761, 1878, 486, 799, 1072, 3047, 809, 2543, 1380, 1936, 2274, 941, 2692, 2212, 3513, 514, 753, 1468, 2976, 2809, 1633, 1289, 2592, 3639, 276, 3243, 1168, 3686, 2731, 2974, 3615, 524, 435, 3412, 320, 2008, 157, 1515, 2805, 1218, 1844, 1214, 3722, 2978, 3580, 841, 1985, 3695, 749, 1170, 166, 2779, 814, 2021, 2741, 2324, 3013, 612, 686, 1838, 3066, 3627, 1352, 3281, 1412, 2544, 653, 1849, 2950, 380, 2915, 3074, 1464, 2755, 3596, 1511, 2297, 445, 244, 150, 434, 1401, 299, 2441, 3322, 1859, 1323, 2849, 1500, 3472, 2746, 1392, 562, 73, 630, 1743, 3447, 2407, 2620, 3435, 3485, 2015, 3196, 1304, 3647, 2724, 1236, 1115, 3226, 1535, 2741, 1341, 2894, 2333, 2145, 125, 3218, 244, 900, 1525, 2516, 1026, 2557, 2645, 941, 2899, 3535, 3022, 1507, 2004, 1977, 21, 3286, 3527, 1755, 1061, 2059, 1177, 3296, 2167, 2332, 2578, 473, 2590, 1980, 1006, 2989, 3757, 3601, 2838, 1180, 1653, 3188, 2100, 1404, 3716, 1464]
        elif config['dataset'] == 'tweet':
            doc_idx = [604, 662, 227, 173, 423, 393, 246, 527, 82, 575, 4, 423, 228, 194, 726, 396, 510, 83, 60, 363, 649, 659, 558, 501, 740, 670, 514, 282, 774, 661, 441, 542, 622, 382, 738, 54, 584, 511, 704, 433, 633, 402, 580, 118, 393, 371, 13, 54, 558, 468, 386, 715, 629, 1, 603, 294, 584, 190, 188, 26, 311, 510, 168, 625, 228, 701, 259, 425, 618, 242, 79, 309, 646, 156, 638, 334, 357, 499, 487, 149, 487, 480, 559, 606, 357, 392, 330, 400, 380, 707, 411, 183, 605, 229, 601, 477, 206, 463, 327, 452, 80, 724, 300, 36, 36, 436, 158, 408, 650, 422, 653, 56, 406, 700, 502, 406, 336, 267, 420, 617, 119, 371, 345, 748, 86, 563, 22, 536, 323, 179, 375, 287, 649, 743, 604, 319, 423, 99, 333, 305, 221, 729, 797, 665, 694, 270, 141, 247, 645, 501, 170, 291, 599, 655, 147, 55, 525, 370, 392, 536, 686, 311, 532, 426, 183, 363, 282, 157, 596, 422, 566, 105, 137, 800, 267, 56, 152, 676, 285, 367, 238, 502, 555, 493, 121, 622, 235, 643, 92, 355, 369, 74, 464, 592, 739, 242, 130, 606, 740, 277]
    elif config['vocabulary_size'] == 2000:
        if config['dataset'] == 'agnews':
            doc_idx = [11474, 17169, 4118, 24477, 13485, 15536, 14771, 14175, 6101, 21725, 17450, 2619, 21842, 4709, 12891, 14016, 4597, 2974, 2139, 7949, 7502, 20363, 20528, 3951, 4068, 4504, 13716, 8775, 5968, 5323, 15620, 7842, 13056, 16784, 24786, 21104, 3756, 13132, 17328, 15658, 2647, 18786, 24868, 24839, 175, 24319, 5147, 20471, 1177, 606, 10171, 11499, 12128, 19889, 20957, 24622, 17535, 9049, 8733, 25237, 18788, 4798, 22745, 20086, 4587, 16091, 3447, 15419, 4333, 21807, 21986, 16959, 13108, 1122, 7478, 14687, 19878, 2434, 17966, 14588, 21535, 23304, 2284, 20633, 4554, 7122, 9717, 2737, 15956, 21097, 10002, 10812, 20723, 1619, 6713, 8271, 10329, 20341, 2240, 1867, 20825, 14584, 8617, 11438, 14138, 23121, 5531, 24340, 14400, 4389, 3288, 14705, 874, 7559, 22890, 23339, 17798, 14492, 25286, 18424, 11040, 1943, 11918, 14503, 17184, 8516, 9841, 25235, 20086, 3091, 18554, 13204, 1836, 13654, 13668, 2911, 10175, 436, 11950, 12959, 11850, 5136, 19639, 1296, 7853, 3213, 8704, 8410, 17060, 13820, 8213, 6505, 4392, 19575, 7341, 23595, 4474, 172, 16257, 22203, 11192, 21676, 375, 7949, 15141, 20049, 5655, 7905, 5284, 8435, 1675, 21830, 20178, 22157, 24656, 14493, 20379, 23611, 15022, 16875, 22582, 6760, 6954, 6114, 23999, 19842, 5896, 642, 17968, 5980, 1908, 1163, 1686, 6522, 1347, 14988, 10578, 19683, 9149, 4852]
        elif config['dataset'] == '20news':
            doc_idx = [2002, 1097, 3697, 1648, 547, 2072, 3486, 2443, 3633, 3159, 2804, 99, 2517, 2424, 2629, 469, 1755, 2015, 3011, 940, 394, 1251, 382, 3704, 544, 1345, 3704, 2508, 1014, 1020, 3544, 2005, 643, 415, 1910, 1722, 2456, 2591, 313, 2283, 2101, 2029, 562, 3735, 45, 2627, 569, 189, 3404, 2290, 2681, 1984, 1953, 3307, 3695, 2879, 1657, 3461, 2385, 1424, 3537, 3584, 2413, 3353, 1839, 1317, 94, 896, 2219, 1271, 2959, 2125, 1881, 349, 2953, 2608, 1548, 1003, 2330, 2555, 1870, 647, 943, 2515, 1961, 2853, 3609, 770, 2547, 1434, 3172, 2647, 161, 3381, 1971, 3254, 2617, 675, 1676, 331, 2541, 1473, 2269, 3550, 3499, 3323, 3655, 2513, 3216, 3158, 2910, 126, 122, 1783, 527, 2459, 306, 474, 507, 2853, 3403, 64, 3, 654, 676, 3681, 1441, 1952, 2480, 1705, 1197, 1514, 2315, 2550, 607, 782, 2781, 2518, 3462, 1753, 435, 1586, 3012, 2251, 2918, 118, 1464, 3318, 3537, 1076, 3191, 669, 2854, 270, 692, 1313, 152, 2189, 1080, 1376, 3402, 367, 201, 3558, 2235, 788, 1281, 231, 1405, 1675, 3108, 1926, 455, 1048, 3430, 2601, 3289, 279, 1388, 3044, 1831, 2937, 342, 635, 1669, 2653, 3070, 671, 3006, 1920, 1490, 3201, 2258, 1943, 472, 1740, 2098, 2124, 1053, 2753]
        elif config['dataset'] == 'tweet':
            doc_idx = [174, 455, 466, 95, 252, 576, 629, 532, 260, 586, 477, 54, 95, 486, 97, 625, 453, 360, 603, 492, 707, 342, 273, 236, 461, 289, 53, 684, 519, 378, 381, 593, 568, 124, 330, 630, 187, 178, 157, 378, 709, 666, 390, 66, 122, 337, 58, 545, 222, 421, 682, 451, 285, 695, 500, 516, 224, 43, 376, 422, 431, 38, 303, 628, 141, 191, 81, 471, 86, 649, 266, 313, 657, 221, 309, 443, 551, 181, 314, 160, 452, 15, 676, 318, 255, 569, 439, 148, 569, 600, 430, 551, 414, 534, 710, 687, 168, 251, 184, 473, 655, 182, 486, 576, 557, 91, 12, 49, 122, 448, 714, 625, 180, 635, 422, 617, 44, 630, 184, 326, 244, 500, 363, 640, 587, 638, 583, 557, 198, 377, 96, 98, 533, 565, 426, 488, 601, 458, 488, 330, 338, 440, 157, 368, 171, 195, 312, 261, 563, 5, 440, 363, 549, 557, 443, 184, 582, 662, 245, 451, 343, 494, 391, 622, 575, 105, 369, 368, 99, 180, 204, 678, 182, 38, 493, 616, 350, 94, 547, 392, 457, 403, 291, 389, 385, 142, 605, 184, 599, 151, 459, 248, 335, 516, 710, 524, 507, 28, 592, 129]
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
