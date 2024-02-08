import pandas as pd
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string

# Load transcript data
df = pd.read_csv("transcripts.csv")


# Preprocess text data
def preprocess_text(text):
    text = remove_stopwords(text, stopwords.words("english"))  # Remove stop words
    text = preprocess_string(text)  # Apply NLTK preprocessing steps
    return text


df["text"] = df["text"].apply(preprocess_text)

# Choose vectorization method (e.g., TF-IDF or Word2Vec)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
transcript_vectors = vectorizer.fit_transform(df["text"])

from hdbscan import HDBSCAN

# Initialize HDBSCAN model
clusterer = HDBSCAN(min_samples=10, min_cluster_size=3)

# Perform clustering
clusters = clusterer.fit_predict(transcript_vectors)


def query_cluster_relevance(query, transcript_vectors, clusters):
    query_vector = vectorizer.transform([query])  # Vectorize query
    similarities = np.dot(query_vector, transcript_vectors.T)  # Calculate similarities
    top_clusters = np.argsort(similarities, axis=1)[
        :, -3:
    ]  # Get top 3 most relevant clusters
    return top_clusters


# Example usage
query = "Can you summarize the discussion about remote control colors?"
top_clusters = query_cluster_relevance(query, transcript_vectors, clusters)
print(f"Most relevant clusters for query: {top_clusters}")



# Design a data pipeline for real-time processing
# Consider using libraries like Kafka or RabbitMQ for message queues
# Update clusters incrementally using techniques like partial updates
# Explore cloud-based solutions for scalability

# Example using a loop (not real-time):
new_transcript = ...  # Get new transcript data
new_vector = vectorizer.transform([new_transcript])
new_cluster = clusterer.partial_fit(new_vector)
if new_cluster != -1:  # New cluster formed
    # Update relevant existing clusters or update visualization


import streamlit as st

# Build Streamlit app with input fields, cluster visualization, and results display
st.title("Meeting Transcript Cluster Search")

query = st.text_input("Enter your query:")

if query:
    top_clusters = query_cluster_relevance(query, transcript_vectors, clusters)
    st.subheader("Most relevant clusters:")
    st.write(top_clusters)

