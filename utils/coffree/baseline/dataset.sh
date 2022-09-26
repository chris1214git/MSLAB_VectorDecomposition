python lstm_baseline.py --encoder=mpnet --dataset=20news | tee record/EXP_20news.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=agnews | tee record/EXP_agnews.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=IMDB | tee record/EXP_IMDB.txt