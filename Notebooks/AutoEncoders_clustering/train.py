from dec import DeepEmbeddingClustering
import numpy as np
import pickle
import config
import utils
import os


if __name__ == "__main__":

    args = config.get_args()

    # embedding vector data
    data_path = 'C:/Users/madhu/Desktop/SeattleUniversity/capstone/resonate_madhu/Resonate/Notebooks/AE_clustering/summary_new_data-embedding.pkl'

    if os.path.isfile(data_path):
        data = pickle.load(open(data_path, "rb"))
        embedding_vector, text, label = data

        # normalize
        normalized_doc_embeddings = utils.normalization_vector(embedding_vector)
        print(normalized_doc_embeddings)
        # check accuracy utilized k-mean
        print(f"length {len(normalized_doc_embeddings)}")
        
        utils.check_kmean_accuracy(normalized_doc_embeddings, label)
        dec = DeepEmbeddingClustering(args, n_clusters=len(np.unique(label)))

        # greedy-layer wise auto-encoder
        dec.initialize(args,
                       normalized_doc_embeddings,text,
                       finetune_iters=args.finetune_iters,
                       layerwise_pretrain_iters=args.layerwise_pretrain_iters)

        # update z space of patent document vector
        dec.cluster(args,text, x_data=normalized_doc_embeddings, y_data=label, test=args.task)

    else:

        print("embedding patent document first!")




