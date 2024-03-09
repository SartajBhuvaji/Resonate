import pandas as pd
import torch
from scipy.io import savemat
import openai
#from openai import OpenAI
from graph_filters import graph_filtering
import scipy.io
from scipy.io import loadmat
import numpy as np
from time import time
from sklearn.cluster import MeanShift,estimate_bandwidth
import matplotlib.pyplot as plt
import openai
import joblib
from langchain.embeddings.openai import OpenAIEmbeddings
from scipy.spatial.distance import euclidean
import os
from uuid import uuid4
import SemanticSearch

#Adding new document/transcript
'''This function will perform two task:
1. embedding on entire data, abstract_data.csv (appending on mat file was causing failure)
2. save embeddings in cluster_data-embedding.mat in format uuid,text
'''
def create_embedding():
    data=pd.read_csv('data/text/abstract_summary_data.csv')
    text, id = data['text'], data['uuid']
    # embedding model
    model_name = 'text-embedding-3-large'
    embed = OpenAIEmbeddings(
        model=model_name,  
        openai_api_key='' #openai_api_key
    )
    embeddings=embed.embed_documents(text)
    savemat('data/embeddings/cluster_data-embedding.mat',{'uuid':id,'text':embeddings})
    

'''Form clusters 
This function will performed tasks:
1. call embedding function
2. form clusters using cluste_datar-embedding.mat file
3. Save predicted labels in cluster_data.csv
'''
def create_Cluster(new_data):
    create_embedding()
    data = loadmat('data/embeddings/cluster_data-embedding.mat')
    features1 = data['text']
    id = data['uuid'].reshape(-1)
    method='dgc'
    features = graph_filtering(features1, method='dgc')
    ibandwidth = estimate_bandwidth(features, quantile=0.30,random_state=42)
    msclustering= MeanShift(bandwidth=ibandwidth,max_iter=900)
    model_path = f'data/models/{method}_model.joblib'
    joblib.dump(msclustering, model_path)
    Z= msclustering.fit_predict(features)  
    print("Clustering:Done")
    
    #add_abstractFile(new_data)
    df = pd.read_csv(f'data/text/abstract_summary_data.csv')
    df['cluster']=Z
    #new_data=pd.DataFrame({'uuid':df['uuid'],'text':df['text'],'PredictedLabel':Z})
    df.to_csv("data/text/cluster_data.csv")
    print('Upadted csv file:Done')

#query
#index = FaissIdx(model) 
def uuid_for_query(query,index):
    
    #predict the cluster label for query
    query_cluster_label=index.search_query(query)
    print(f"label pred: {query_cluster_label[0]}")
    df=pd.read_csv('data/text/cluster_data.csv')
    #match all uuid from predicted label
    filtered_uuids = df[df['PredictedLabel'] == query_cluster_label[0]]['uuid'].tolist()
    return filtered_uuids

def initialize_FAISS():
    model =SemanticSearch.SemanticEmbedding()
    index= SemanticSearch.FaissForQuerySearch(model) 
    data = pd.read_csv('data/text/cluster_data.csv')
    features1 = data['text']
    uuids=data['uuid']
    labels=data['PredictedLabel']
    for text,uuid,label in zip(features1,uuids,labels):
        index.add_summary(text,uuid,label)
    return index
