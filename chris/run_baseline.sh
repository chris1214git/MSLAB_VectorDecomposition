# BCE
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 3

# ListNet
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 3

# MSE
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 3

# topk_MSE
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion topk_MSE --n_time 3
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion topk_MSE --n_time 3
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion topk_MSE --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion topk_MSE --n_time 3

# MultiLabelMarginLossCustom
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MultiLabelMarginLossCustom --n_time 3
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MultiLabelMarginLossCustom --n_time 3
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MultiLabelMarginLossCustom --n_time 3
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MultiLabelMarginLossCustom --n_time 3

# preprocess baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2

# cross domain
python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
