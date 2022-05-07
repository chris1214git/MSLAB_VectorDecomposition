# EXP 1 SBERT embedding on different datasets
python lstm_baseline.py --encoder=mpnet --dataset=20news | tee record/EXP_1_20news.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=agnews | tee record/EXP_1_agnews.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=IMDB | tee record/EXP_1_IMDB.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=wiki | tee record/EXP_1_wiki.txt &&

# EXP 3 On 20news with TFIDF different embedding algorithm
python lstm_baseline.py --dataset=20news --encoder=bert | tee record/EXP_3_Bert.txt &&
python lstm_baseline.py --dataset=20news --encoder=mpnet | tee record/EXP_3_SBERT.txt &&
python lstm_baseline.py --dataset=20news --encoder=average | tee record/EXP_3_Glove.txt &&
python lstm_baseline.py --dataset=20news --encoder=doc2vec | tee record/EXP_3_doc2vec.txt