import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
parser.add_argument('--experiment', type=str, default='search')
args = parser.parse_args()
config = vars(args)

if config['experiment'] == 'search':
    print('--- search parameters ---')
    batch_size_list = [8, 16, 32]
    architecture_list = ['before', 'after']
    activation_list = ['sigmoid', 'tanh']  
    lr_list = [2e-3, 1e-2, 1e-3, 1e-4] # 2e-3 is ZTM default lr
    weight_decay_list = [1e-1, 1e-2, 1e-3]
    

    comb = list(product(batch_size_list, architecture_list, activation_list, lr_list, weight_decay_list))
    for parameters in comb:
        batch_size, architecture, activation, lr, weight_decay = parameters
        cmd = f"python3 ztm_decoder.py --dataset {config['dataset']} --architecture {architecture} --activation {activation} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
        os.system(cmd)
elif config['experiment'] == 'dataset':
    print('--- run all dataset ---')
    dataset_list = ['20news', 'agnews', 'IMDB']
    architecture_list = ['before', 'after']
    activation = 'sigmoid'
    batch_size = 16
    lr = 2e-3
    weight_decay = 1e-1

    comb = list(product(architecture_list, dataset_list))
    for parameters in comb:
        architecture, dataset = parameters
        cmd = f"python3 ztm_decoder.py --dataset {dataset} --architecture {architecture} --activation {activation} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
        os.system(cmd)
else:
    print('--- experiment ---')
    activation_list = ['sigmoid', 'tanh']
    architecture_list = ['after', 'before']
    batch_size = 8
    lr = 2e-3
    weight_decay = 1e-1

    comb = list(product(activation_list, architecture_list))
    for parameters in comb:
        activation, architecture = parameters
        cmd = f"python3 ztm_decoder.py --dataset {config['dataset']} --architecture {architecture} --activation {activation} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
        os.system(cmd)
