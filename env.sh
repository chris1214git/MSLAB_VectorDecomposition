wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d ./data
rm glove.6B.zip
unzip data/IMDB.zip -d ./data

cd data
python3 ./get_docembedding.py