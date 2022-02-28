import os
import re
import torch
import numpy as np
from math import log
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    os.system('rm -f tmp')
    print('Using cuda {} for training...'.format(int(np.argmax(memory_available))))
    return "cuda:{}".format(int(np.argmax(memory_available)))


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def show_settings(config):
    print('-------- Info ---------')
    settings = ""
    for key in list(config.keys()):
        settings += "{}: {}\n".format(key, config.get(key))
    print(settings)
    print('-----------------------')


def split_data(dataset, config):
    train_length = int(len(dataset)*0.6)
    valid_length = int(len(dataset)*0.2)
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, lengths=[train_length, valid_length, test_length],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, valid_loader, test_loader

def doc_filter(raw_document, vocab):
    PATTERN = r"(?u)\b\w\w+\b"
    doc = re.findall(PATTERN, raw_document.lower())
    return [x for x in doc if x in vocab]

def generate_graph(doc_list, word2index, index2word):
    window_size = 10
    windows = []

    # Traverse Each Document & Move window on each of them
    for doc in doc_list:
        length = len(doc)
        if length <= window_size:
            windows.append(doc)
        else:
            for i in range(length-window_size+1):
                window = doc[i: i+window_size]
                windows.append(window)
    
    word_freq = {}
    word_pair_count = {}
    for window in tqdm(windows, desc='Calculate word pair: '):
        appeared = set()
        for i in range(len(window)):
            if window[i] not in appeared:
                if window[i] in word_freq:
                    word_freq[window[i]] += 1
                else:
                    word_freq[window[i]] = 1
                appeared.add(window[i])
            if i != 0:
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word2index[word_i]
                    word_j = window[j]
                    word_j_id = word2index[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
    
    row = []
    col = []
    edge = []
    weight = []
    # pmi as weights

    num_window = len(windows)
    # count_mean = np.array(list(word_pair_count.values())).mean()
    for key in tqdm(word_pair_count, desc='Construct Edge: '):
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_freq[index2word[i]]
        word_freq_j = word_freq[index2word[j]]
        pmi = log((1.0 * count / num_window) /
                (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        if count >= 15:
            edge.append([i, j])
            edge.append([j, i])
        weight.append(pmi)

    print('# of Node: {}\n# of Edge: {}'.format(len(word2index), len(edge)))

    return edge