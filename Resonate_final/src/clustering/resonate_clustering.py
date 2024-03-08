import json
import os

import joblib
import numpy as np
import pandas as pd

import scipy.sparse as sp
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from scipy.io import loadmat, savemat
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import kneighbors_graph

# import resonate_semantic_search as SemanticSearch

import src.clustering.resonate_semantic_search as SemanticSearch


def normalize_adj(adj, lmbda=1):
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
    # Use a context manager to ensure the file is properly closed after opening
    with open(json_file_path, "r") as file:
        # Load the JSON data
        data = json.load(file)

    return data


# Adding new document/transcript
"""This function will perform two task:
1. embedding on entire data, abstract_data.csv (appending on mat file was causing failure)
2. save embeddings in cluster-embedding.mat in format uuid,text
"""


class Clustering:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.method = "dgc"

        if not os.path.exists("./data/clusteringFiles/cluster_data.csv"):
            self.create_Cluster()

        self.index = self.initialize_FAISS()

    def create_embedding(self):
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

    """Form clusters
    This function will performed tasks:
    1. call embedding function
    2. form clusters using cluster-embedding.mat file
    3. Save predicted labels in cluster_data.csv
    """

    def create_Cluster(self):
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
        # predict the cluster label for query
        query_cluster_label = self.index.search_query(query)
        print(f"label pred: {query_cluster_label[0]}")
        df = pd.read_csv("./data/clusteringFiles/cluster_data.csv")
        # match all uuid from predicted label
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
