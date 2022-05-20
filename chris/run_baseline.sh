# # BCE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1

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

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 1

# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --n_epoch 50

# # Loss
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE2 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE3 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet2 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet3 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet4 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet4 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.2 --save_dir dropout

# # Loss
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet4 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.2 --save_dir dropout

# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet2 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet3 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet4 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.2 --save_dir dropout
# python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.2 --save_dir dropout

# # BCE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1

# # ListNet
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1

# # MSE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion MSE --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1

# # MSE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion MSE2 --n_time 1
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion MSE2 --n_time 1
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion MSE2 --n_time 1
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE2 --n_time 1

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE2 --n_time 1
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion MSE2 --n_time 1
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion MSE2 --n_time 1

# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf-gensim --criterion MSE2 --n_time 1 --n_epoch 50

# # cross domain
# python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1
# python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1

# python3 baseline.py --dataset 20news --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1
# python3 baseline.py --dataset 20news --dataset2 wiki --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1

# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1 --n_epoch 40

# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1 --n_epoch 40

# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 1 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 1 --n_epoch 40

# # BCE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 1 --preprocess_config_dir parameters_baseline3

# # ListNet
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 1 --preprocess_config_dir parameters_baseline3

# # MSE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 1 --preprocess_config_dir parameters_baseline3

# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 1 --n_epoch 50 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 1 --n_epoch 50 --preprocess_config_dir parameters_baseline3
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 1 --n_epoch 50 --preprocess_config_dir parameters_baseline3

# # BCE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion BCE --n_time 10
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion BCE --n_time 10
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion BCE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 10

# # ListNet
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 10

# # MSE
# python3 baseline.py --dataset 20news --model_name average --label_type tf-idf --criterion MSE --n_time 10
# python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf --criterion MSE --n_time 10
# python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf --criterion MSE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10
# python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10
# python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 10

# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10 --n_epoch 50
# python3 baseline.py --dataset wiki --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10 --n_epoch 50


# 0519 baseline
# ListNet
python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion ListNet_sigmoid_L1 --n_time 10

python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion MSE --n_time 10

python3 baseline.py --dataset 20news --model_name average --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name doc2vec --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name bert --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset agnews --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset IMDB --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 10
python3 baseline.py --dataset 20news --model_name mpnet --label_type yake --criterion BCE --n_time 10

# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion BCE --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 agnews --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 20news --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10 --n_epoch 40
# python3 baseline.py --dataset wiki --dataset2 IMDB --model_name mpnet --label_type tf-idf-gensim --criterion MSE --n_time 10 --n_epoch 40

# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --train_size 0.1 --save_dir train_size_01 --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --train_size 0.3 --save_dir train_size_03 --n_time 10
# python3 baseline.py --dataset 20news --model_name mpnet --label_type tf-idf-gensim --criterion ListNet_sigmoid_L1 --train_size 0.5 --save_dir train_size_05 --n_time 10