# Description:Creates clusters based on the uploaded transcripts and returns the uuid of the documents that are similar to the query.
# Reference Code: https://github.com/chakib401/smoothing_sentence_embeddings/blob/master/utils.py

'''
Paper Citation for def normalize_adj(): 
Fettal, Chakib, Lazhar Labiod, and Mohamed Nadif. 
"More Discriminative Sentence Embeddings via Semantic Graph Smoothing." 
arXiv preprint arXiv:2402.12890 (2024).
'''

import json
import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import src.clustering.resonate_semantic_search as SemanticSearch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from scipy.io import loadmat, savemat
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import kneighbors_graph

def normalize_adj(adj, lmbda=1):
    '''
    Normalize adjacency matrix of semantic graph
    '''
    adj = adj + lmbda * sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)

    adj = adj + lmbda * sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def graph_filtering(features, degree=2, lmbda=1, nn=10, alpha=0.5, t=5, method="sgc"):
    """
    This function will perform graph filtering based on four polynomial filters
    We keep n=10, as per paper. And is used to calculate the graph (adjacency matrix)
    between 10 vectors/features.

    **That is why we have 10 pre-existing transcripts placed in pinecone (through the ont_time_script)
    **If you want to change the number of transcripts, you will have to change the number of neighbors
    """
    adj = kneighbors_graph(features, n_neighbors=nn, metric="cosine")
    adj = (adj + adj.T) / 2

    S = normalize_adj(adj, lmbda)
    xx = features
    yy = features.copy()
    if method in ["sgc", "s2gc"]:
        for _ in range(degree):
            xx = S @ xx
            yy += xx
        if method == "sgc":
            return xx
        elif method == "s2gc":
            return yy
    elif method == "appnp":
        for _ in range(degree):
            xx = (1 - alpha) * S @ xx + alpha * features
        return xx
    elif method == "dgc":
        k = degree + 1
        for _ in range(1, degree + 1):
            xx = (1 - t / k) * xx + (t / k) * (S @ xx)
        return xx
    else:
        raise "unrecognized filter"


def load_json_config(json_file_path="./config/config.json"):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


class Clustering:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.method = "dgc"

        if not os.path.exists("./data/clusteringFiles/cluster_data.csv"):
            self.create_Cluster()

        self.index = self.initialize_FAISS()

    def create_embedding(self):
        '''This function will perform two task:
        1. embedding on entire data, abstract_data.csv
        2. save embeddings in cluster_data-embedding.mat in format uuid, text
        '''
        data = pd.read_csv("./data/summaryFiles/abstract_summary_data.csv")
        json_config = load_json_config()

        text, id = data["text"], data["uuid"]
        # embedding model
        embed = OpenAIEmbeddings(
            model=json_config["EMBEDDING_MODEL_NAME"], openai_api_key=self.api_key
        )
        embeddings = embed.embed_documents(text)
        savemat(
            "./data/embeddingFiles/cluster-embedding.mat",
            {"uuid": id, "text": embeddings},
        )


    def create_Cluster(self):
        '''
        This function will perform following tasks:
        1. call embedding function
        2. form clusters using cluste_data-embedding.mat file
        3. Save predicted labels in cluster_data.csv
        '''
        self.create_embedding()

        data = loadmat("./data/embeddingFiles/cluster-embedding.mat")
        features1 = data["text"]

        features = graph_filtering(features1, method=self.method)
        ibandwidth = estimate_bandwidth(features, quantile=0.30, random_state=42)
        msclustering = MeanShift(bandwidth=ibandwidth, max_iter=900)
        msclustering.fit(features)
        model_path = f"./data/clusteringFiles/{self.method}_model.joblib"
        joblib.dump(msclustering, model_path)

        print("Model saved")

        df = pd.read_csv(f"./data/summaryFiles/abstract_summary_data.csv")
        df["cluster"] = msclustering.predict(features)
        df.to_csv("./data/clusteringFiles/cluster_data.csv")
        print("Cluster data saved")
        self.index = self.initialize_FAISS()



    def uuid_for_query(self, query):
        '''
        Returns the uuids of the documents that are similar to the query, based on the clustering
        '''
        query_cluster_label = self.index.search_query(query)
        print(f"Predicted Label : {query_cluster_label[0]}")
        df = pd.read_csv("./data/clusteringFiles/cluster_data.csv")
        filtered_uuids = df[df["cluster"] == query_cluster_label[0]]["uuid"].tolist()
        return filtered_uuids

    def initialize_FAISS(self):
        model = SemanticSearch.SemanticEmbedding()
        index = SemanticSearch.FaissForQuerySearch(model)
        data = pd.read_csv("./data/clusteringFiles/cluster_data.csv")
        features1 = data["text"]
        uuids = data["uuid"]
        labels = data["cluster"]
        for text, uuid, label in zip(features1, uuids, labels):
            index.add_summary(text, uuid, label)
        return index


if __name__ == "__main__":
    load_dotenv("./config/.env")
    Clustering_obj = Clustering()
    print(
        Clustering_obj.uuid_for_query(
            "What is the goal of defining maintainability for the new diffs architecture?"
        )
    )
    print(
        Clustering_obj.uuid_for_query(
            "What was the design component for remote control?"
        )
    )
