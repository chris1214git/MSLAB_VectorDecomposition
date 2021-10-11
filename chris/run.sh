# python3 unsupervised_docvec_decompose.py --dataset PubMed --normalize_word_embedding --topk_word_freq_threshold 0 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset PubMed --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset PubMed --normalize_word_embedding --topk_word_freq_threshold 300 --document_vector_agg_weight 'IDF'

# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 0 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 300 --document_vector_agg_weight 'IDF'

# python3 unsupervised_docvec_decompose.py --dataset IMDB --normalize_word_embedding --topk_word_freq_threshold 0 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset IMDB --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF'
# python3 unsupervised_docvec_decompose.py --dataset IMDB --normalize_word_embedding --topk_word_freq_threshold 300 --document_vector_agg_weight 'IDF'

# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'mean'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'uniform'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'gaussian'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'exponential'
# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'pmi'

# python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF' --embedding_file glove.6B.100d.txt
python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF' --embedding_file glove.6B.200d.txt
python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF' --embedding_file glove.6B.300d.txt
python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF' --embedding_file glove.6B.50d.txt
python3 unsupervised_docvec_decompose.py --dataset CNN --normalize_word_embedding --topk_word_freq_threshold 100 --document_vector_agg_weight 'IDF' --embedding_file wiki-news-300d-1M.vec


