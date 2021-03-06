# no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline --no_keyword

python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline2 --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline2 --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline2 --no_keyword

# origin
python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline

python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline2
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline2
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline2

# wiki
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline2 --no_keyword
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline2

# 2 gram
python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline --ngram 2
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline --ngram 2
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline --ngram 2

python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline2 --ngram 2
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline2 --ngram 2
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline2 --ngram 2

python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline --ngram 2
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline2 --ngram 2

# parameters_baseline3
python3 keyword_extract_v2.py --ngram 1 --dataset 20news --preprocess_config_dir parameters_baseline3
python3 keyword_extract_v2.py --ngram 1 --dataset agnews --preprocess_config_dir parameters_baseline3
python3 keyword_extract_v2.py --ngram 1 --dataset IMDB --preprocess_config_dir parameters_baseline3
python3 keyword_extract_v2.py --ngram 1 --dataset wiki --preprocess_config_dir parameters_baseline3

