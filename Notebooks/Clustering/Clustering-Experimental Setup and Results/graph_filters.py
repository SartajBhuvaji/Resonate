from sklearn.neighbors import kneighbors_graph
from utils import normalize_adj


def graph_filtering(features, degree=2, lmbda=1, nn=10, alpha=.5, t=5,method='sgc'):
    adj = kneighbors_graph(features, n_neighbors=nn, metric='cosine')
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
    elif method == 'appnp':
        for _ in range(degree):
            xx = (1-alpha) * S @ xx + alpha * features
        return xx
    elif method == 'dgc':
        k = degree + 1
        for _ in range(1, degree+1):
            xx = (1 - t / k) * xx + (t / k) * (S @ xx)
        return xx
    else:
        raise 'unrecognized filter'

