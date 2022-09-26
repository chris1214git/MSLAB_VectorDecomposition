python lstm_baseline.py --dataset=20news --encoder=bert | tee record/EXP_Bert.txt &&
python lstm_baseline.py --dataset=20news --encoder=mpnet | tee record/EXP_SBERT.txt &&
python lstm_baseline.py --dataset=20news --encoder=average | tee record/EXP_Glove.txt &&
python lstm_baseline.py --dataset=20news --encoder=doc2vec | tee record/EXP_doc2vec.txt