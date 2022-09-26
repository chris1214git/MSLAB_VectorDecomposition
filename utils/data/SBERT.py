import os
import sys
from datasets import load_dataset
import numpy as np

documents = load_dataset("imdb",split="train+test")["text"]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2',device="cuda:3")

embeddings = model.encode(documents,show_progress_bar=True,batch_size=256)
np.save("docvec_IMDB_SBERT_768d",embeddings)