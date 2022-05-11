import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
parser.add_argument('--experiment', type=str, default='search')
parser.add_argument('--crossdomain', action="store_true")
args = parser.parse_args()
config = vars(args)
if not config['crossdomain']:
    if config['experiment'] == 'search':
        print('--- search parameters ---')
        batch_size_list = [8, 16, 32]
        activation_list = ['sigmoid', 'tanh']  
        lr_list = [2e-3, 1e-2, 1e-3, 1e-4] # 2e-3 is ZTM default lr
        weight_decay = 0

        comb = list(product(batch_size_list, activation_list, lr_list))
        for parameters in comb:
            batch_size, activation, lr = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {config['dataset']}  --activation {activation} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
            os.system(cmd)
    elif config['experiment'] == 'check':
        print('--- run all dataset & encoder ---')
        dataset_list = ['20news', 'agnews', 'IMDB', 'wiki']
        encoder_list = ['mpnet', 'bert', 'doc2vec', 'average']
        loss_list = ['listnet', 'mse']
        activation = 'sigmoid'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0

        comb = list(product(dataset_list, encoder_list, loss_list))
        for parameters in comb:
            dataset, encoder, loss = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {dataset} --activation {activation} --encoder {encoder} --lr {lr} --loss {loss} --weight_decay {weight_decay} --batch_size {batch_size}"
            os.system(cmd)
    elif config['experiment'] == 'check_news':
        print('--- run all 20news & agnews ---')
        dataset_list = ['20news', 'agnews']
        encoder_list = ['mpnet', 'bert', 'doc2vec', 'average']
        loss_list = ['listnet', 'mse']
        activation = 'sigmoid'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0

        comb = list(product(dataset_list, encoder_list, loss_list))
        for parameters in comb:
            dataset, encoder, loss = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {dataset} --activation {activation} --encoder {encoder} --lr {lr} --loss {loss} --weight_decay {weight_decay} --batch_size {batch_size}"
            os.system(cmd)
    elif config['experiment'] == 'check_others':
        print('--- run all IMDB & wiki ---')
        dataset_list = ['IMDB', 'wiki']
        encoder_list = ['mpnet', 'bert', 'doc2vec', 'average']
        loss_list = ['listnet', 'mse']
        activation = 'sigmoid'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0

        comb = list(product(dataset_list, encoder_list, loss_list))
        for parameters in comb:
            dataset, encoder, loss = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {dataset} --activation {activation} --encoder {encoder} --lr {lr} --loss {loss} --weight_decay {weight_decay} --batch_size {batch_size}"
            os.system(cmd)
    else:
        print('--- experiment ---')
        activation_list = ['sigmoid', 'tanh']
        architecture_list = ['after', 'before']
        batch_size = 8
        lr = 2e-3
        weight_decay = 0

        comb = list(product(activation_list, architecture_list))
        for parameters in comb:
            activation, architecture = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {config['dataset']} --architecture {architecture} --activation {activation} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size}"
            os.system(cmd)
else:
    print('--- Cross Domain ---')
    dataset2_list = ['20news', 'agnews', 'IMDB']
    epochs = 300
    batch_size = 16
    lr = 2e-3
    weight_decay = 0

    comb = dataset2_list#list(product(activation_list, architecture_list))
    for parameters in comb:
        dataset2 = parameters
        config['experiment'] = 'CrossDomain_wiki_' + dataset2 
        cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {'wiki'} --dataset2 {dataset2} --architecture {'concatenate'}  --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {epochs}"
        os.system(cmd)