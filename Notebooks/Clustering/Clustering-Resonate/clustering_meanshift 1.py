import os
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from graph_filters import graph_filtering
from sklearn.metrics.cluster import adjusted_rand_score as ari
import joblib


def load_data(dataset):
    data = loadmat(f'data/embeddings/{dataset}-embedding.mat')
    features = data['x']
    labels = data['y'].reshape(-1)
    return features, labels

def calculate_cluster_centroids(features, Z):
    cluster_centroids = []
    for cluster_label in np.unique(Z):
        cluster_points = features[Z == cluster_label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        cluster_centroids.append(cluster_centroid.flatten())
    return cluster_centroids

def calculate_distances(query_embedding, cluster_centroids):
    return [np.linalg.norm(query_embedding - centroid.reshape(-1)) for centroid in cluster_centroids]

def perform_meanshift_clustering(features):
    ibandwidth = estimate_bandwidth(features, quantile=0.30, random_state=42)
    msclustering = MeanShift(bandwidth=ibandwidth, max_iter=900)
    Z = msclustering.fit_predict(features)
    return Z

def plot_pca_clusters(features, Z, method):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    label_dict = {0: 'social media', 1: 'office relocation', 2: 'remote control', 3: 'architecture'}

    plt.figure(figsize=(9, 4))
    for i in range(len(np.unique(Z))):
        plt.scatter(features_pca[Z == i, 0], features_pca[Z == i, 1], label=label_dict[i])
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.title(f'PCA Visualization of predicted Clusters for {method}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('predicted_clusters.png')

def smoothing_meanshift(dataset, method):
    features, labels = load_data(dataset)
    features1 = features 
    if method:
        features = graph_filtering(features1, method=method)

    Z = perform_meanshift_clustering(features)
    raw_data = pd.read_csv(f'data/text/{dataset}.csv')
    text = raw_data['text']
    data = {'text': text, 'ActualLabel': labels, 'PredictedLabel': Z}
    pred_df = pd.DataFrame(data)
    pred_df.to_csv(f"C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/Clustering-meeting-TESTING/Clustering-meeting-TESTING/data/Smoothing-meanshift/{method}/smms-{method}.csv")

    cluster_centroids = calculate_cluster_centroids(features, Z)
    query_embedding = loadmat(f'C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/Clustering-meeting-TESTING/Clustering-meeting-TESTING/data/embeddings/social_media_3.mat')['x']
    distances = calculate_distances(query_embedding, cluster_centroids)
    closest_cluster = np.argmin(distances)
    plot_pca_clusters(features, Z, method)

 
    return closest_cluster

def load_or_train_model(dataset, method):
    model_path = f'/data/models/{method}_model.joblib'
    if os.path.exists(model_path):
        # Load the existing model
        msclustering = joblib.load(model_path)
    else:
        # Train a new model
        features, labels = load_data(dataset)
        features1 = features 
        if method:
            features = graph_filtering(features1, method=method)

        ibandwidth = estimate_bandwidth(features, quantile=0.30, random_state=42)
        msclustering = MeanShift(bandwidth=ibandwidth, max_iter=900)
        msclustering.fit(features)

        # Save the trained model
        joblib.dump(msclustering, model_path)

    return msclustering

def update_model(dataset, method):
    # Load or train the clustering model
    msclustering = load_or_train_model(dataset, method)

    # Perform clustering on the new data
    features, labels = load_data(dataset)
    Z = msclustering.predict(features)

    # Save the updated model
    model_path = f'C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/Clustering-meeting-TESTING/Clustering-meeting-TESTING/data/models/{method}_model.joblib'
    joblib.dump(msclustering, model_path)

    return Z

closest_cluster = smoothing_meanshift('test2', 'dgc')
print(f"closest cluster to the query is {closest_cluster}")

new_Z = update_model('new_meeting_data', 'dgc')
print("Model updated with new data.")
