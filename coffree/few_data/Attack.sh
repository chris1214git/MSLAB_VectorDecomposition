mkdir -p record
# Inductive
python attack_baseline.py --dataset=20news --ratio=0.1 --inductive=True | tee record/Attack_ind_20news_0.1.txt &&
python attack_baseline.py --dataset=20news --ratio=0.06 --inductive=True | tee record/Attack_ind_20news_0.06.txt &&
python attack_baseline.py --dataset=20news --ratio=0.8 --inductive=True | tee record/Attack_ind_20news_0.8.txt &&
# Transductive
python attack_baseline.py --dataset=20news --ratio=0.1 | tee record/Attack_trans_20news_0.1.txt &&
python attack_baseline.py --dataset=20news --ratio=0.06 | tee record/Attack_trans_20news_0.06.txt &&
python attack_baseline.py --dataset=20news --ratio=0.8 | tee record/Attack_trans_20news_0.8.txt