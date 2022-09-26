import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
parser.add_argument('--experiment', type=str, default='Unlabeled_cos')
args = parser.parse_args()
config = vars(args)

if config['experiment'] == 'Unlabeled_cos':
    print('--- search parameters ---')
    batch_size_list = [8, 16, 32] 
    lr_list = [2e-3, 1e-2, 1e-3, 1e-4, 2e-5, 1e-5] # 2e-3 is ZTM default lr
    weight_decay = 0

    comb = list(product(batch_size_list, lr_list))
    for parameters in comb:
        batch_size, lr = parameters
        cmd = f"python3 ide_semi.py --experiment {config['experiment']} --dataset {config['dataset']} --lr {lr} --batch_size {batch_size}"
        os.system(cmd)
elif config['experiment'] == 'unlabeled_decoder':
    print('--- search parameters of unlabeled_decoder ---')
    ae_epochs_list = [10, 15, 20, 25, 30] 
    ae_lr_list = [2e-3, 1e-3, 1e-4, 2e-5]
    lr_list = [2e-3, 1e-3, 1e-4, 2e-5]
    weight_decay = 0

    comb = list(product(ae_epochs_list, ae_lr_list, lr_list))
    for parameters in comb:
        ae_epochs, ae_lr, lr = parameters
        cmd = f"python3 ide_semi.py --experiment {config['experiment']} --dataset {config['dataset']} --lr {lr} --epochs {500} --ae_lr {ae_lr} --ae_epochs {ae_epochs}"
        os.system(cmd)