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
    
    elif config['experiment'] == 'p_value_20news':
        print('[p-value] Fix 20news')
        dataset = '20news'
        encoder_list = ['bert', 'doc2vec', 'average']
        architecture = 'concatenate'
        target = 'tf-idf-gensim'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio = 0.5

        comb = list(product(encoder_list, seed_list))
        for parameters in comb:
            encoder, seed = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {300} --seed {seed}"
            os.system(cmd)
    
    elif config['experiment'] == 'p_value_mpnet':
        print('[p-value] Fix mpnet')
        dataset_list = ['agnews', 'IMDB', 'wiki']
        encoder = 'mpnet'
        architecture = 'concatenate'
        target = 'tf-idf-gensim'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio = 0.5

        comb = list(product(dataset_list, seed_list))
        for parameters in comb:
            dataset, seed = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {300} --seed {seed}"
            os.system(cmd)
    elif config['experiment'] == 'p_value_target_20news':
        print('[p-value] Fix 20news & mpnet')
        dataset = '20news'
        encoder = 'mpnet'
        architecture = 'concatenate'
        target_list = ['keybert', 'yake', 'tf-idf']
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio = 0.5

        comb = list(product(target_list, seed_list))
        for parameters in comb:
            target, seed = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {300} --seed {seed}"
            os.system(cmd)
    
    elif config['experiment'] == 'p_value_target_agnews':
        print('[p-value] Fix agnews & mpnet')
        dataset = 'agnews'
        encoder = 'mpnet'
        architecture = 'concatenate'
        target_list = ['keybert', 'yake']
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio = 0.8

        comb = list(product(target_list, seed_list))
        for parameters in comb:
            target, seed = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {300} --seed {seed}"
            os.system(cmd)

    elif config['experiment'] == 'p_value_target_IMDB':
        print('[p-value] Fix IMDB & mpnet')
        dataset = 'IMDB'
        encoder = 'mpnet'
        architecture = 'concatenate'
        target_list = ['keybert', 'yake']
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio = 0.8

        comb = list(product(target_list, seed_list))
        for parameters in comb:
            target, seed = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {300} --seed {seed}"
            os.system(cmd)

    elif config['experiment'] == 'p_value_train_ratio':
        print('[p-value] Fix 20news & mpnet, but training ratio')
        dataset = '20news'
        encoder_list = ['mpnet', 'bert', 'doc2vec', 'average']
        architecture = 'concatenate'
        target = 'tf-idf-gensim'
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed_list = [123, 234, 345, 456, 567, 678, 789, 890, 910, 911]
        ratio_list = [0.5, 0.3, 0.1]

        comb = list(product(encoder_list, seed_list, ratio_list))
        for parameters in comb:
            encoder, seed, ratio = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {250} --seed {seed} --ratio {ratio}"
            os.system(cmd)

    elif config['experiment'] == 'visualize':
        print('[p-value] Generate Document')
        dataset_list = ['20news', 'agnews', 'IMDB']
        encoder = 'mpnet'
        architecture = 'concatenate'
        target_list = ['yake', 'keybert', 'tf-idf-gensim']
        batch_size = 16
        lr = 2e-3
        weight_decay = 0
        seed = 123
        ratio = 0.8

        comb = list(product(dataset_list, target_list))
        for parameters in comb:
            dataset, target = parameters
            cmd = f"python3 ide_topic.py --experiment {config['experiment']+str(int(ratio*100))+'_'+str(seed)} --dataset {dataset} --architecture {architecture} --encoder {encoder} --target {target} --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {50} --seed {seed} --ratio {ratio} --check_document {True}"
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
    epochs = 50
    batch_size = 16
    lr = 2e-3
    weight_decay = 0

    comb = dataset2_list#list(product(activation_list, architecture_list))
    for parameters in comb:
        dataset2 = parameters
        config['experiment'] = 'CrossDomain_wiki_' + dataset2 
        cmd = f"python3 ide_topic.py --experiment {config['experiment']} --dataset {'wiki'} --dataset2 {dataset2} --architecture {'concatenate'}  --lr {lr} --weight_decay {weight_decay} --batch_size {batch_size} --epochs {epochs}"
        os.system(cmd)
