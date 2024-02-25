import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


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
