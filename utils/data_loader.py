import os
import re
import numpy as np

from tqdm.auto import tqdm
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset


def load_word2emb(embedding_file):
    word2embedding = dict()
    word_dim = int(re.findall(r".(\d+)d", embedding_file)[0])

    with open(embedding_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split()
            word = line[0]
            embedding = list(map(float, line[1:]))
            word2embedding[word] = np.array(embedding)

    print("Number of words:%d" % len(word2embedding))

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
    else:
        raise NotImplementedError

    return {"documents": documents, "target": target, "num_classes": num_classes}
