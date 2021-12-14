import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    os.system('rm -f tmp')
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
