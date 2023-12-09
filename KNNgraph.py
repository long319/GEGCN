import os
import pdb
import time
import os.path
import scipy.io as scio
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
from torch_sparse.tensor import SparseTensor
from sklearn.metrics.pairwise import cosine_similarity as cos

def load_adj(features, normalization=True, normalization_type='normalize',                                #加载邻接矩阵
              k_nearest_neighobrs=10, prunning_one=False, prunning_two=False , common_neighbors=2):
    if normalization:
        if normalization_type == 'minmax_scale':
            features = minmax_scale(features)
        elif normalization_type == 'maxabs_scale':
            features = maxabs_scale(features)
        elif normalization_type == 'normalize':
            features = normalize(features)
        elif normalization_type == 'robust_scale':
            features = robust_scale(features)
        elif normalization_type == 'scale':
            features = scale(features)
        elif normalization_type == '255':
            features = np.divide(features, 255.)
        elif normalization_type == '50':
            features = np.divide(features, 50.)
        else:
            print("Please enter a correct normalization type!")
            pdb.set_trace()
    print(features)
    # construct three kinds of adjacency matrix

    adj, adj_wave, adj_hat = construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one,
                                                        prunning_two, common_neighbors)
    return adj, adj_hat

def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()


    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    # adj_hat 就是D-1/2AD-1/2
    adj_hat = construct_adjacency_hat(adj)  # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    # print("The construction of adjacency matrix is finished!")
    # print("The time cost of construction: ", time.time() - start_time)

    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix

########################################################################################################
# 返回稀疏张量，似乎这个函数没有用到
def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                        torch.FloatTensor(three_tuple[1]),
                                        torch.Size(three_tuple[2]))
    return sparse_tensor

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape


def load_KNN_graph(args):
    k = args.k
    data = scio.loadmat('./dataset/' + args.dataset_name + '.mat')
    features = data['X']     # <class 'numpy.ndarray'>
    # features = features.numpy()
    save_direction = os.path.join(os.path.join("./dataset/", args.dataset_name), "knn_graph.npy")

    if os.path.exists(save_direction):
        adj = np.load(save_direction, allow_pickle = True)
        print("Successfully load KNN graph...")
    else:
        # construct three kinds of adjacency matrix
        print("Constructing the KNN graph of " + args.dataset_name)
        adj, adj_hat = load_adj(features, k_nearest_neighobrs = k)
        adj = adj.todense()
        print(adj)
        # save these scale and matrix
        # print("Saving the adjacency matrix to " + save_direction)
        # np.save(os.path.join(save_direction, 'adj'), adj_list)
        np.save(save_direction, adj)

        print("Construction completed..." )
    adj = torch.from_numpy(adj).float().to_sparse()
    row, col = adj._indices()
    val = adj._values()
    size = adj.size()
    return SparseTensor(row = row, col = col, value = val, sparse_sizes = size)

###################################################################################################################

'''
KNN Graph generated by AM-GCN
'''

def construct_graph(dataset, topk):
    if not os.path.exists('./dataset/' + dataset + '/knn/'):
        os.makedirs('./dataset/' + dataset + '/knn/')
    fname = './dataset/' + dataset + '/knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    data = scio.loadmat('./dataset/' + dataset + '.mat')
    features = data['X']  # <class 'numpy.ndarray'
    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]   # 获得前TOPK个最相似的值
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))  # 写入文件
    f.close()

def generate_knn(args):
    if not os.path.exists('./dataset/' + args.dataset_name + '/knn/c' + str(args.k) + '.txt'):
        topk = args.k
        construct_graph(args.dataset_name, topk)
        f1 = open('./dataset/' + args.dataset_name + '/knn/tmp.txt', 'r')
        f2 = open('./dataset/' + args.dataset_name + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

def knn_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx   # D-1A

def knn_load_graph_v2(args, n):  # n 我猜测是数据集的节点数
    featuregraph_path = './dataset/' + args.dataset_name + '/knn/c' + str(args.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)     # 保证是对称矩阵
    nfadj = knn_normalize(fadj + sp.eye(fadj.shape[0])).todense()
    nfadj = torch.from_numpy(nfadj).float().to_sparse()
    row, col = nfadj._indices()
    val = nfadj._values()
    size = nfadj.size()


    return SparseTensor(row = row, col = col, value = val, sparse_sizes = size)