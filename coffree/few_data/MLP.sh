mkdir -p record
python mlp_baseline.py --dataset=20news --ratio=0.1 | tee record/MLP_20news_0.1.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.08 | tee record/MLP_20news_0.08.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.06 | tee record/MLP_20news_0.06.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.8 | tee record/MLP_20news_0.8.txt