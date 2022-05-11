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

# Loss
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE2 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MSE3 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet2 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet3 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet4 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset 20news --model_name mpnet --label_type keybert --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.2 --save_dir dropout

python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion BCE --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE2 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MSE3 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet2 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet3 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet4 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion ListNet_sigmoid_L1 --n_time 1 --dropout 0.2 --save_dir dropout
python3 baseline.py --dataset agnews --model_name mpnet --label_type keybert --criterion MultiLabelMarginLoss --n_time 1 --dropout 0.2 --save_dir dropout