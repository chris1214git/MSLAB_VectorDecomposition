import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
parser.add_argument('--encoder', type=str, default="all")
args = parser.parse_args()
config = vars(args)

activation_list = ['sigmoid', 'tanh']
if config['encoder'] == 'all':
    encoder_list = ['mpnet', 'bert', 'average', 'doc2vec']
elif config['encoder'] == 'transformer':
    encoder_list = ['mpnet', 'bert']
else:
    encoder_list = ['average', 'doc2vec']
lr_list = [2e-3, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4] # 2e-3 is ZTM original lr
weight_decay_list = [0, 1e-1, 1e-2, 1e-3]
batch_size_list = [8, 16, 32]

comb = list(product(activation_list, encoder_list, lr_list, weight_decay_list, batch_size_list))
for parameters in comb:
    activation, encoder, lr, weight_decay, batch_size = parameters
    cmd = f"python3 ztm_decoder.py --dataset {config['dataset']} --activation {activation} --encoder {encoder} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
    os.system(cmd)