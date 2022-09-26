# dataset
python3 unsupervised_docvec_decompose.py --dataset CNN 
python3 unsupervised_docvec_decompose.py --dataset IMDB 
python3 unsupervised_docvec_decompose.py --dataset PubMed

# distribution
python3 unsupervised_docvec_decompose.py --dataset CNN --document_vector_agg_weight mean --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --document_vector_agg_weight uniform --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --document_vector_agg_weight gaussian --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --document_vector_agg_weight exponential --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --document_vector_agg_weight pmi --n_document 5000
# distribution
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight uniform --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight gaussian --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight exponential --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight pmi --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000

# embedding_file
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file glove.6B.100d.txt --n_document 10000
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file doc2vecC.imdb.100d.txt --n_document 10000
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file fasttext.en.100d.txt --n_document 10000

python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file glove.6B.100d.txt --n_document 10000
python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file doc2vecC.imdb.100d.txt --n_document 10000
python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file fasttext.en.100d.txt --n_document 10000

python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file glove.6B.100d.txt
python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file doc2vecC.imdb.100d.txt
python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file fasttext.en.100d.txt

# embedding_file
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file doc2vecC.imdb.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.02 --embedding_file fasttext.en.200d.txt --n_document 5000

python3 unsupervised_docvec_decompose.py --dataset IMDB --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset IMDB --lr 0.02 --embedding_file doc2vecC.imdb.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset IMDB --lr 0.02 --embedding_file fasttext.en.200d.txt --n_document 5000

python3 unsupervised_docvec_decompose.py --dataset PubMed --lr 0.02 --embedding_file glove.6B.200d.txt
python3 unsupervised_docvec_decompose.py --dataset PubMed --lr 0.02 --embedding_file doc2vecC.imdb.200d.txt
python3 unsupervised_docvec_decompose.py --dataset PubMed --lr 0.02 --embedding_file fasttext.en.200d.txt


# dim
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file glove.6B.50d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --embedding_file glove.6B.300d.txt --n_document 5000

python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file glove.6B.50d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset IMDB --embedding_file glove.6B.300d.txt --n_document 5000

python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file glove.6B.50d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset PubMed --embedding_file glove.6B.300d.txt --n_document 5000



# dataset
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt
# distribution
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight uniform --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight gaussian --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight exponential --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight pmi --n_document 5000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 5000
# embedding_file
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 1000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 1000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.00001 --bpdn_epochs 400 --embedding_file doc2vecC.imdb.200d.txt --n_document 1000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.0000001 --bpdn_epochs 400 --embedding_file fasttext.en.200d.txt --n_document 1000
python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.005 --lasso_epochs 2000 --lasso_alphal 0.0001 --bpdn_epochs 1000 --embedding_file textgcn.20ng.200d.txt --n_document 1000
# # embedding_file(KNN)
# python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 1000 --no_sklearn --no_bp
# python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 1000 --no_sklearn --no_bp
# python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.00001 --bpdn_epochs 400 --embedding_file doc2vecC.imdb.200d.txt --n_document 1000 --no_sklearn --no_bp
# python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.0000001 --bpdn_epochs 400 --embedding_file fasttext.en.200d.txt --n_document 1000 --no_sklearn --no_bp
# python3 unsupervised_docvec_decompose.py --dataset CNN --epochs 2000 --momentum 0.9995 --lr 0.005 --lasso_epochs 2000 --lasso_alphal 0.0001 --bpdn_epochs 1000 --embedding_file textgcn.20ng.200d.txt --n_document 1000 --no_sklearn --no_bp
# embedding_file
python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.00001 --bpdn_epochs 400 --embedding_file doc2vecC.imdb.200d.txt --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.0000001 --bpdn_epochs 400 --embedding_file fasttext.en.200d.txt --n_document 2000
# python3 unsupervised_docvec_decompose.py --dataset IMDB --epochs 2000 --momentum 0.9995 --lr 0.005 --lasso_epochs 2000 --lasso_alphal 0.0001 --bpdn_epochs 1000 --embedding_file textgcn.imdb.200d.txt --n_document 2000

python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --lasso_epochs 2000 --bpdn_epochs 400 --lr 0.02 --embedding_file glove.6B.200d.txt --document_vector_agg_weight mean --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.00001 --bpdn_epochs 400 --embedding_file doc2vecC.PubMed.200d.txt --n_document 2000
python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --momentum 0.9995 --lr 0.03 --lasso_epochs 2000 --lasso_alphal 0.0000001 --bpdn_epochs 400 --embedding_file fasttext.en.200d.txt --n_document 2000
# python3 unsupervised_docvec_decompose.py --dataset PubMed --epochs 2000 --momentum 0.9995 --lr 0.005 --lasso_epochs 2000 --lasso_alphal 0.0001 --bpdn_epochs 1000 --embedding_file textgcn.PubMed.200d.txt --n_document 2000
