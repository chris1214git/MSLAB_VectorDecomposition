python mlp_baseline.py --dataset=20news --ratio=0.1 | tee record/MLP_20news_0.1.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.2 | tee record/MLP_20news_0.2.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.4 | tee record/MLP_20news_0.4.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.6 | tee record/MLP_20news_0.6.txt &&
python mlp_baseline.py --dataset=20news --ratio=0.8 | tee record/MLP_20news_0.8.txt