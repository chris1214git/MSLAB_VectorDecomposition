# # BCE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 1

# # ListNet
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 1

# # MSE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 1

# MultiLabelMarginLossCustomV
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MultiLabelMarginLossCustomV:1 --n_time 1
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MultiLabelMarginLossCustomV:10 --n_time 1
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MultiLabelMarginLossCustomV:100 --n_time 1

# preprocess baseline2
# BCE
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2

# ListNet
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2

# MSE
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2

python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline2

# # cross domain
# python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 3 --preprocess_config_dir parameters_baseline2

# python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 3 --preprocess_config_dir parameters_baseline2

# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline2

# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline2
