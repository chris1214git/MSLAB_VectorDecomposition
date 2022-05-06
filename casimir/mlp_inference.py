import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--crossdomain', action="store_true")
args = parser.parse_args()
config = vars(args)
if not config['crossdomain']:
    print('[INFO] Single Dataset Experiment')
    dataset_list = ['20news', 'agnews', 'IMDB', 'wiki']
    encoder_list = ['mpnet', 'bert', 'doc2vec', 'average']
    loss_list = ['bce', 'mse', 'listnet']

    comb = list(product(dataset_list, encoder_list, loss_list))
    for parameters in comb:
        dataset, encoder, loss = parameters
        cmd = f"python3 mlp_baseline.py --dataset {dataset} --encoder {encoder} --loss {loss}"
        os.system(cmd)
else:
    print('[INFO] Cross Domain Experiment')
    dataset_list = ['20news', 'wiki']
    dataset2_list = ['20news', 'agnews', 'IMDB', 'wiki']
    loss_list = ['bce', 'mse', 'listnet']
    comb = list(product(dataset_list, dataset2_list, loss_list))
    for parameters in comb:
        dataset, dataset2, loss = parameters
        if dataset != dataset2:
            cmd = f"python3 mlp_baseline.py --dataset {dataset} --dataset2 {dataset2} --loss {loss} --batch_size {16}"
            os.system(cmd)
