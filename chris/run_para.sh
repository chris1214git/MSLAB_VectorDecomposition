# default
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0. --L1 0 --w_sum_reg 0 --w_clip_value -10
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.01 --momentum 0. --L1 0 --w_sum_reg 0 --w_clip_value -10

# momentum
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0.9 --L1 0 --w_sum_reg 0 --w_clip_value -10
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0.99 --L1 0 --w_sum_reg 0 --w_clip_value -10
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0.999 --L1 0 --w_sum_reg 0 --w_clip_value -10
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0.9999 --L1 0 --w_sum_reg 0 --w_clip_value -10

# clip
python3 unsupervised_docvec_decompose.py --dataset CNN --lr 0.1 --momentum 0.999 --L1 0 --w_sum_reg 0 --w_clip_value 0
python3 unsupervised_docvec_decompose.py --dataset CNN --w_clip_value -10

# L1
python3 unsupervised_docvec_decompose.py --dataset CNN --L1 0
python3 unsupervised_docvec_decompose.py --dataset CNN --L1 1e-4
python3 unsupervised_docvec_decompose.py --dataset CNN --L1 1e-5
python3 unsupervised_docvec_decompose.py --dataset CNN --L1 1e-6
python3 unsupervised_docvec_decompose.py --dataset CNN --L1 1e-7

# w_sum_reg
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg 1e-2
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg 1e-3
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg 1e-4
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg 1e-5

# w_sum_reg_mul
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg_mul 0.8
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg_mul 0.9
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg_mul 1
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg_mul 1.1
python3 unsupervised_docvec_decompose.py --dataset CNN --w_sum_reg_mul 1.2


