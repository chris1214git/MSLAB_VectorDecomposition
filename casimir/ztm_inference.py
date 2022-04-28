import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
args = parser.parse_args()
config = vars(args)

activation_list = ['sigmoid', 'tanh']
encoder_list = ['mpnet', 'bert', 'average', 'roberta']
lr_list = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
weight_decay_list = [0, 1e-1, 1e-2, 1e-3]
batch_size_list = [8, 16, 32]

comb = list(product(activation_list, encoder_list, lr_list, weight_decay_list, batch_size_list))
for parameters in comb:
    activation, encoder, lr, weight_decay, batch_size = parameters
    cmd = f"python3 ztm_decoder.py --dataset {config['dataset']} --activation {activation} --encoder {encoder} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
    os.system(cmd)