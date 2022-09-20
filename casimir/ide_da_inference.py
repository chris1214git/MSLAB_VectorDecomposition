import os
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='document decomposition.')
parser.add_argument('--dataset', type=str, default="20news")
args = parser.parse_args()
config = vars(args)

