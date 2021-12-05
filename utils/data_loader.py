import os
import re
import numpy as np

from tqdm.auto import tqdm
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset


def normalize_wordemb(word2embedding):
    # Every word emb should have norm 1
    word_emb = []
    word_list = []
    for word, emb in word2embedding.items():
        word_list.append(word)
        word_emb.append(emb)

    word_emb = np.array(word_emb)

    for i in range(len(word_emb)):
        norm = np.linalg.norm(word_emb[i])
        word_emb[i] = word_emb[i] / norm

    for word, emb in tqdm(zip(word_list, word_emb)):
        word2embedding[word] = emb

    return word2embedding


def load_word2emb(embedding_file):
    word2embedding = dict()

    with open(embedding_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            word2embedding[word] = np.array(embedding)

    print("Number of words:%d" % len(word2embedding))

    return word2embedding


def load_word2embedding(word2embedding_path: str, word2embedding_normalize: bool):
    word2embedding = None
    if word2embedding_path != '':
        assert os.path.exists(word2embedding_path)
        print("Loading word2embedding from {}".format(word2embedding_path))
        word2embedding = load_word2emb(word2embedding_path)
        if word2embedding_normalize:
            print("Normalizing word2embedding")
            word2embedding = normalize_wordemb(word2embedding)

    return word2embedding


def load_document(dataset):
    if dataset == "20news":
        num_classes = 20
        raw_text, target = fetch_20newsgroups(data_home="./", subset='all', categories=None,
                                              shuffle=False, return_X_y=True)
        documents = [doc.strip("\n") for doc in raw_text]
        target = list(target)
    elif dataset == "IMDB":
        data = load_dataset("imdb",split="train+test")
        documents = data["text"]
        target = data["label"]
        num_classes = 2

    elif dataset == "MR":
        target = []
        documents = []
        num_classes = 2

        sub_file = ["rt-polarity.pos", "rt-polarity.neg"]
        dir_prefix = "./SentEval/data/downstream/MR"
        for target_type in sub_file:
            file_name = os.path.join(dir_prefix, target_type)
            with open(file_name, "r") as f:
                context = f.readlines()
                documents.extend(context)

            # assign label
            label = 1 if target_type == "rt-polarity.pos" else 0
            label = [label] * len(context)
            target.extend(label)
    elif dataset == "CR":
        target = []
        documents = []
        num_classes = 2

        sub_file = ["custrev.pos", "custrev.neg"]
        dir_prefix = "./SentEval/data/downstream/CR"
        for target_type in sub_file:
            file_name = os.path.join(dir_prefix, target_type)
            with open(file_name, "r") as f:
                context = f.readlines()
                documents.extend(context)

            # assign label
            label = 1 if target_type == "custrev.pos" else 0
            label = [label] * len(context)
            target.extend(label)
    elif dataset == "SUBJ":
        target = []
        documents = []
        num_classes = 2

        sub_file = ["subj.objective", "subj.subjective"]
        dir_prefix = "./SentEval/data/downstream/SUBJ"
        for target_type in sub_file:
            file_name = os.path.join(dir_prefix, target_type)
            with open(file_name, "r") as f:
                context = f.readlines()
                documents.extend(context)

            # assign label
            label = 1 if target_type == "subj.objective" else 0
            label = [label] * len(context)
            target.extend(label)
    else:
        raise NotImplementedError

    return {"documents": documents, "target": target, "num_classes": num_classes}
