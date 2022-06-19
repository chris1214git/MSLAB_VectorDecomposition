# Cross domain
mkdir -p record &&
python lstm_baseline.py --encoder=mpnet --dataset=wiki --dataset2=20news | tee record/EXP_cross_20news.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=wiki --dataset2=IMDB | tee record/EXP_cross_IMDB.txt &&
python lstm_baseline.py --encoder=mpnet --dataset=wiki --dataset2=agnews | tee record/EXP_cross_agnews.txt

# python lstm_baseline.py --encoder=mpnet --dataset=20news | tee record/EXP_20news.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=agnews | tee record/EXP_agnews.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=IMDB | tee record/EXP_IMDB.txt &&
# python lstm_baseline.py --dataset=20news --encoder=bert | tee record/EXP_Bert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet | tee record/EXP_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=average | tee record/EXP_Glove.txt &&
# python lstm_baseline.py --dataset=20news --encoder=doc2vec | tee record/EXP_doc2vec.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=keybert | tee record/EXP_Keybert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=yake | tee record/EXP_Yake.txt

# EXP 1 SBERT embedding on different datasets
# python lstm_baseline.py --encoder=mpnet --dataset=20news | tee record/EXP_1_20news.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=agnews | tee record/EXP_1_agnews.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=IMDB | tee record/EXP_1_IMDB.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=wiki | tee record/EXP_1_wiki.txt &&
# EXP 2 SBERT on 20news with different target
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=keybert | tee record/EXP_2_Keybert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=yake | tee record/EXP_2_Yake.txt
# EXP 3 On 20news with TFIDF different embedding algorithm
# python lstm_baseline.py --dataset=20news --encoder=bert | tee record/EXP_3_Bert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet | tee record/EXP_3_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=average | tee record/EXP_3_Glove.txt &&
# python lstm_baseline.py --dataset=20news --encoder=doc2vec | tee record/EXP_3_doc2vec.txt
# EXP 4 preprocess: 1/100 different embedding
# python lstm_baseline.py --dataset=20news --encoder=bert --min_df=0.01 | tee record/EXP_4_Bert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --min_df=0.01 | tee record/EXP_4_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=average --min_df=0.01 | tee record/EXP_4_Glove.txt &&
# python lstm_baseline.py --dataset=20news --encoder=doc2vec --min_df=0.01 | tee record/EXP_4_doc2vec.txt &&
# EXP 5 preprocess: 1/100 SBERT on different datasets
# python lstm_baseline.py --encoder=mpnet --dataset=20news --min_df=0.01 | tee record/EXP_5_20news.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=agnews --min_df=0.01 | tee record/EXP_5_agnews.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=IMDB --min_df=0.01 | tee record/EXP_5_IMDB.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=wiki --min_df=0.01 | tee record/EXP_5_wiki.txt
# EXP 6 min_df=20, different embedding algorithm
# python lstm_baseline.py --dataset=20news --encoder=bert --min_df=20 | tee record/EXP_6_Bert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --min_df=20 | tee record/EXP_6_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=average --min_df=20 | tee record/EXP_6_Glove.txt &&
# python lstm_baseline.py --dataset=20news --encoder=doc2vec --min_df=20 | tee record/EXP_6_doc2vec.txt &&
# EXP 7 min_df=20, different dataset
# python lstm_baseline.py --encoder=mpnet --dataset=20news --min_df=20 | tee record/EXP_7_20news.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=agnews --min_df=20 | tee record/EXP_7_agnews.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=IMDB --min_df=20 | tee record/EXP_7_IMDB.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=wiki --min_df=20 | tee record/EXP_7_wiki.txt &&
# EXP 8 min_df=20, different target
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=tf-idf-gensim --min_df=20 | tee record/EXP_8_TFIDF_raw.txt
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=keybert --min_df=62 | tee record/EXP_8_Keybert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=yake --min_df=62 | tee record/EXP_8_Yake.txt
# EXP 9 min_df=1/300, different embedding train_size=0.5
# python lstm_baseline.py --dataset=20news --encoder=mpnet --target=tf-idf --min_df=0.003 | tee record/EXP_TFIDF_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=bert --min_df=0.003 | tee record/EXP_9_Bert.txt &&
# python lstm_baseline.py --dataset=20news --encoder=mpnet --min_df=0.003 | tee record/EXP_9_SBERT.txt &&
# python lstm_baseline.py --dataset=20news --encoder=average --min_df=0.003 | tee record/EXP_9_Glove.txt &&
# python lstm_baseline.py --dataset=20news --encoder=doc2vec --min_df=0.003 | tee record/EXP_9_doc2vec.txt &&
# EXP 10 min_df=1/300, different dataset train_size=0.5
# python lstm_baseline.py --encoder=mpnet --dataset=20news --min_df=0.003 | tee record/EXP_10_20news.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=agnews --min_df=0.003 | tee record/EXP_10_agnews.txt &&
# python lstm_baseline.py --encoder=mpnet --dataset=IMDB --min_df=0.003 | tee record/EXP_10_IMDB.txt