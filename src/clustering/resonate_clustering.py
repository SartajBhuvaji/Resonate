import os
from time import time
from uuid import uuid4

# from openai import OpenAI
import joblib

# import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import scipy.io
import torch
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from scipy.io import loadmat, savemat
from scipy.spatial.distance import euclidean
from sklearn.cluster import MeanShift, estimate_bandwidth

# from openai import OpenAI
from src.clustering.graph_filters import graph_filtering
import json


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


def create_embedding(api_key):
    data = pd.read_csv("./data/summaryFiles/abstract_summary_data.csv")

    json_config = load_json_config()

    text, id = data["text"], data["meeting_uuid"]
    # embedding model
    model_name = json_config["EMBEDDING_MODEL"]
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    embeddings = embed.embed_documents(text)
    savemat(
        "./data/embeddingFiles/cluster-embedding.mat",
        {"meeting_uuid": id, "text": embeddings},
    )


"""Form clusters
This function will performed tasks:
1. call embedding function
2. form clusters using cluster-embedding.mat file
3. Save predicted labels in cluster_data.csv
"""


def create_Cluster(api_key):
    create_embedding(api_key)
    # create_embedding_single(new_data)

    data = loadmat("./data/embeddingFiles/cluster-embedding.mat")
    features1 = data["text"]
    id = data["meeting_uuid"].reshape(-1)
    method = "dgc"
    features = graph_filtering(features1, method="dgc")
    ibandwidth = estimate_bandwidth(features, quantile=0.30, random_state=42)
    msclustering = MeanShift(bandwidth=ibandwidth, max_iter=900)
    model_path = f"./data/clusteringFiles/{method}_model.joblib"
    joblib.dump(msclustering, model_path)
    Z = msclustering.fit_predict(features)
    print("Clustering:Done")

    # add_abstractFile(new_data)
    df = pd.read_csv(f"./data/summaryFiles/abstract_summary_data.csv")
    df["cluster"] = Z
    # new_data=pd.DataFrame({'uuid':df['uuid'],'text':df['text'],'PredictedLabel':Z})
    df.to_csv("./data/clusteringFiles/cluster_data.csv")
    print("Upadted csv file:Done")


# Query
"""loading the store model"""


def load_model():
    method = "dgc"
    model_path = f"./data/clusteringFiles/{method}_model.joblib"
    if os.path.exists(model_path):
        # Load the existing model
        msclustering = joblib.load(model_path)
    else:
        print("Unable to load the model")
    return msclustering


"""Make prediction using saved model"""


def predict_cluster(query):
    # load existing model
    cluster_model = load_model()
    # predict the label
    predicted_label = cluster_model.predict(query)
    return predicted_label


"""create embedding for query"""


def query_embedding(query, api_key):

    json_config = load_json_config()

    model_name = json_config["EMBEDDING_MODEL"]
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    embeddings = embed.embed_documents(query[0])
    return embeddings


"""Extracting the uuids for query"""


def uuid_for_query(query, api_key):
    # get embedding for query
    query_embed = query_embedding(query, api_key)
    # predict the cluster label for query
    query_cluster_label = predict_cluster(query_embed)
    print(f"label pred: {query_cluster_label[0]}")
    df = pd.read_csv("./data/clusteringFiles/cluster_data.csv")
    # match all uuid from predicted label
    filtered_uuids = df[df["cluster"] == query_cluster_label[0]][
        "meeting_uuid"
    ].tolist()
    print(filtered_uuids)
    return filtered_uuids


if __name__ == "__main__":
    load_dotenv("./config/.env")
    create_Cluster(api_key=os.environ.get("OPENAI_API_KEY"))
