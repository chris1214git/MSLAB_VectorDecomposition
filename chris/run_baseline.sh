python baseline.py --dataset 20news --model_name roberta --label_type bow --criterion BCE
python baseline.py --dataset agnews --model_name roberta --label_type bow --criterion BCE
python baseline.py --dataset tweet --model_name roberta --label_type bow --criterion BCE

python baseline.py --dataset 20news --model_name mpnet --label_type bow --criterion BCE
python baseline.py --dataset agnews --model_name mpnet --label_type bow --criterion BCE
python baseline.py --dataset tweet --model_name mpnet --label_type bow --criterion BCE

python baseline.py --dataset 20news --model_name average --label_type bow --criterion BCE
python baseline.py --dataset agnews --model_name average --label_type bow --criterion BCE
python baseline.py --dataset tweet --model_name average --label_type bow --criterion BCE

python baseline.py --dataset 20news --model_name doc2vec --label_type bow --criterion BCE
python baseline.py --dataset agnews --model_name doc2vec --label_type bow --criterion BCE
python baseline.py --dataset tweet --model_name doc2vec --label_type bow --criterion BCE
