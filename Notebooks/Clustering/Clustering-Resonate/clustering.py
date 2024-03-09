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
#from openai import OpenAI
import joblib
from langchain.embeddings.openai import OpenAIEmbeddings
from scipy.spatial.distance import euclidean
import os
from uuid import uuid4

#Adding new document/transcript
'''This function will perform two task:
1. embedding on entire data, abstract_data.csv (appending on mat file was causing failure)
2. save embeddings in cluster-embedding.mat in format uuid,text
'''
def create_embedding():
    data=pd.read_csv('data/text/abstract_summary_data.csv')
    text, id = data['text'], data['uuid']
    # embedding model
    model_name = 'text-embedding-3-large'
    embed = OpenAIEmbeddings(
        model=model_name,  
        openai_api_key='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'
    )
    embeddings=embed.embed_documents(text)
    savemat('data/embeddings/cluster-embedding.mat',{'uuid':id,'text':embeddings})
    

'''Form clusters 
This function will performed tasks:
1. call embedding function
2. form clusters using cluster-embedding.mat file
3. Save predicted labels in cluster_data.csv
'''
def create_Cluster(new_data):
    #add_abstractFile(new_data)
    create_embedding()
    #create_embedding_single(new_data)

    data = loadmat('data/embeddings/cluster-embedding.mat')
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


#Query
'''loading the store model'''
def load_model():
    method='dgc'
    model_path = f'data/models/{method}_model.joblib'
    if os.path.exists(model_path):
        # Load the existing model
        msclustering = joblib.load(model_path)
    else:
        print("Unable to load the model")
    return msclustering


'''Make prediction using saved model'''
def predict_cluster(query):
    #load existing model
    cluster_model=load_model()
    #predict the label
    predicted_label=cluster_model.predict(query)
    return predicted_label

'''create embedding for query'''
def query_embedding(query):
    model_name = 'text-embedding-3-large'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key='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'
    )
    embeddings=embed.embed_documents(query[0])
    return embeddings

'''Extracting the uuids for query'''
def uuid_for_query(query):
    #get embedding for query
    query_embed=query_embedding(query)
    #predict the cluster label for query
    query_cluster_label=predict_cluster(query_embed)
    print(f"label pred: {query_cluster_label[0]}")
    df=pd.read_csv('data/text/cluster_data.csv')
    #match all uuid from predicted label
    filtered_uuids = df[df['cluster'] == query_cluster_label[0]]['uuid'].tolist()
    return filtered_uuids

