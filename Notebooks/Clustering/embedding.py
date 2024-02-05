import pandas as pd
import torch
from scipy.io import savemat
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


for dataset in ['abstract']:#['ag_news', 'dbpedia', 'bbc', 'classic4', 'classic3', 'ohsumed', '20ng', 'R8', 'R52']:
    df = pd.read_csv(f'data/text/{dataset}.csv')
    text, labels = df['text'].values, df['label'].values.reshape(-1)
    embedder = SentenceTransformer('all-mpnet-base-v2')
    embeddings = embedder.encode(text)
    savemat(f'data/embeddings/{dataset}-embedding.mat', {'x': embeddings, 'y': labels})
    print("Embedding:Done")

