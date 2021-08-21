# from tqdm import tqdm
import numpy as np
from tqdm import tqdm

word2embedding = dict()
with open("glove.6B.100d.txt","r") as f:
    for line in f:
        line = line.strip().split()
        word = line[0]
        embedding = list(map(float,line[1:]))
        word2embedding[word] = embedding

print("Number of words:%d" % len(word2embedding))

output_file =  open("docvector.txt","w")
with open("IMDB.txt","r") as f:
    for line in tqdm(f):
        words = line.strip().split()
        emb = np.mean([word2embedding[w] for w in words if w in word2embedding], axis=0)
        emb = " ".join(list(map(str,emb)))
        output_file.write(emb+"\n")
output_file.close()
        
