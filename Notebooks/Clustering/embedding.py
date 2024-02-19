import pandas as pd
import torch
from scipy.io import savemat
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


dataset = 'test2' #abstract.csv
df = pd.read_csv(f'/Users/prachiteechouhan/Documents/CPSC DSP /Topic-based clustering/SentenceTransformer-Smoothing/Clustering-meeting-TESTING/data/text/{dataset}.csv')
text, labels = df['text'].values, df['label'].values.reshape(-1)
embedder = SentenceTransformer('all-mpnet-base-v2') 
embeddings = embedder.encode(text)

savemat(f'/Users/prachiteechouhan/Documents/CPSC DSP /Topic-based clustering/SentenceTransformer-Smoothing/Clustering-meeting-TESTING/data/embeddings/{dataset}-sentencetransformer-embedding.mat', {'x': embeddings, 'y': labels})
print("Embedding:Done")

