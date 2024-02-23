import scipy.sparse as sp
import torch


def normalize_adj(adj, lmbda=1):
    adj = adj + lmbda * sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)

    adj = adj + lmbda * sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_dense(adj, lmbda=1):
    adj = adj + lmbda * sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    mx = adj / d_inv_sqrt / d_inv_sqrt[:, None]

    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def scipy_csr_to_torch_csr(matrix):
    """
    Convert a sparse matrix in SciPy CSR format to Torch CSR format.

    Parameters:
        matrix (scipy.sparse.csr_matrix): The input sparse matrix in SciPy CSR format.

    Returns:
        torch.sparse.FloatTensor: The converted sparse matrix in Torch CSR format.
    """
    if not sp.isspmatrix_csr(matrix):
        raise ValueError("Input matrix must be in CSR format.")

    coo = matrix.tocoo()
    # indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)

    return torch.sparse_csr_tensor(coo.row, coo.col, values, shape)


def torch_csr_to_scipy_csr(matrix):
    """
    Convert a PyTorch CSR matrix to a SciPy CSR matrix.

    Parameters:
        matrix (torch.sparse.FloatTensor): The input sparse matrix in PyTorch CSR format.

    Returns:
        scipy.sparse.csr_matrix: The converted sparse matrix in SciPy CSR format.
    """
    # Get the indices, values, and shape of the PyTorch CSR matrix
    matrix = matrix.to_sparse_coo()
    indices = matrix.indices()
    values = matrix.values()
    shape = matrix.size()

    # Create a SciPy CSR matrix from the indices, values, and shape
    scipy_csr = sp.csr_matrix((values.numpy(), indices.numpy()), shape=shape)

    return scipy_csr


# hyperparameter optimization


import itertools
import numpy as np
import hyperopt
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import Apply


def recursive_find_nodes(root, node_type='switch'):
    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(recursive_find_nodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(recursive_find_nodes(node, node_type))
    elif isinstance(root, (Apply)):
        if root.name == node_type:
            nodes.append(root)

        for node in root.pos_args:
            if node.name == node_type:
                nodes.append(node)
        for _, node in root.named_args:
            if node.name == node_type:
                nodes.append(node)
    return nodes


def parameters(space):
    # Analyze the domain instance to find parameters
    parameters = {}
    if isinstance(space, dict):
        space = list(space.values())
    for node in recursive_find_nodes(space, 'switch'):

        # Find the name of this parameter
        paramNode = node.pos_args[0]
        assert paramNode.name == 'hyperopt_param'
        paramName = paramNode.pos_args[0].obj

        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramName] = np.array(range(len(values)))
    return parameters


def spacesize(space):
    # Compute the number of possible combinations
    params = parameters(space)
    return np.prod([len(values) for values in params.values()])


def suggest(new_ids, domain, trials, seed):

    # Analyze the domain instance to find parameters
    params = parameters(domain.expr)

    # Compute all possible combinations
    s = [[(name, value) for value in values] for name, values in params.items()]
    values = list(itertools.product(*s))

    rval = []
    for _, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs = {name: np.array([new_id]) for name in params.keys()}
        vals = {name: np.array([value]) for name, value in values[new_id]}

        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval
