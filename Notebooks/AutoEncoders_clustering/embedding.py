import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os
from sklearn.preprocessing import LabelEncoder
# Load the dataset
dataset = 'summary_new_data'  # abstract.csv
df = pd.read_csv(f'C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/AE_clustering/data/{dataset}.csv')


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform labels
encoded_labels = label_encoder.fit_transform(df['label'])


text, labels = df['text'].values, encoded_labels.reshape(-1)

# Generate embeddings
embedder = SentenceTransformer('all-mpnet-base-v2')
embeddings = embedder.encode(text)

# Save embeddings and labels using pickle
save_path = 'C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/AE_clustering/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
data_to_pickle = (embeddings, text, labels)
with open(os.path.join(save_path, f'{dataset}-embedding.pkl'), 'wb') as f:
    pickle.dump((data_to_pickle), f)

print("Embedding saved in pickle format.")