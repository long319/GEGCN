"""
@Project   : MvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : Dataloader.py
"""
import os
import pdb
import time
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from args import parameter_parser

def printHomogeneity(adj, labels):
    edges_num = np.count_nonzero(adj)
    count = 0
    indexes = np.nonzero(adj)
    indexes = torch.tensor(indexes)
    for i in range(edges_num):
        indexes_raw = indexes[0][i]
        indexes_columns = indexes[1][i]
        if labels[indexes_raw] == labels[indexes_columns]:
            count = count + 1
    Homogeneity = count / edges_num
    print('Homogeneity is: {:.4f}'.format(Homogeneity))
    return Homogeneity


def load_data_Isogram(args,device):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['feature']
    feature_list = []
    adj_list = []
    labels = data['label']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    idx_labeled_train,idx_labelde_val,idx_unlabeled = train_idx.flatten(),val_idx.flatten(),test_idx.flatten()
    if args.dataset=='ACM3025':
        feature=normalize(features)
        feature = torch.from_numpy(feature).float().to(device)
        adj_list.append(torch.from_numpy(data['PTP']).float())
        adj_list.append(torch.from_numpy(data['PLP']).float())
        adj_list.append(torch.from_numpy(data['PAP']).float())
        feature_list.append(feature)
        labels=torch.tensor(labels.argmax(1))
    if args.dataset == 'imdb5k':
        feature = torch.from_numpy(features).float().to(device)
        adj_list.append(torch.from_numpy(data['MAM']).float())
        adj_list.append(torch.from_numpy(data['MDM']).float())
        adj_list.append(torch.from_numpy(data['MYM']).float())
        feature_list.append(feature)
        labels = torch.tensor(labels.argmax(1))
    return feature_list, adj_list, labels, list(idx_labeled_train), list(idx_unlabeled)

def load_data(args, device):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    try:
        features = data['X']
    except:
        features1 = data['X1']
        features2 = data['X2']
        features = [[]]
        features[0].append(features1)
        features[0].append(features2)
        print(features[0])
    feature_list = []
    adj_list = []
    try:
        labels = data['truth'].flatten()
    except:
        labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.ratio, seed=args.shuffle_seed)
    labels = torch.from_numpy(labels).long()
    num_classes = len(np.unique(labels))

    labels_one_hot = torch.eye(num_classes)[labels, :]
    labels_one_hot_mask = torch.zeros_like(labels_one_hot)
    labels_one_hot_mask[idx_labeled, :] = labels_one_hot[idx_labeled, :]
    Avg_Homogeneity = []
    for i in range(features.shape[1]):
        # print("Loading the data of" + str(i) + "th view")
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        direction_judge = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading the adjacency matrix of " + str(i) + "th view of " + args.dataset)
            adj = torch.from_numpy(ss.load_npz(direction_judge).todense()).float().to(device)
            Avg_Homogeneity.append(printHomogeneity(adj.cpu().detach().numpy(),labels))
        else:
            print("Constructing the adjacency matrix of " + str(i) + "th view of " + args.dataset)
            adj = construct_adjacency_matrix(feature, args.knns, args.pr1, args.pr2, args.common_neighbors)
            adj = ss.coo_matrix(adj)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            # adj = construct_laplacian(adj)
            save_direction = './adj_matrix/' + args.dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the adjacency matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz', adj)
            adj = torch.from_numpy(adj.todense()).float().to(device)

        feature = torch.from_numpy(feature).float().to(device)
        # lp = construct_sparse_float_tensor(lp).to(device)
        feature_list.append(feature)
        adj_list.append(adj)
    print('Avg_Homogeneity is: {:.4f}'.format(np.mean(Avg_Homogeneity)))
        # adj_splist.append(adj_sp)
    return feature_list, adj_list, labels, idx_labeled, idx_unlabeled


def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_construct = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if prunning_one:
        # Pruning strategy 1
        original_adj = adj.A
        judges_matrix = original_adj == original_adj.T
        adj = original_adj * judges_matrix
        adj = ss.csc_matrix(adj)
    # obtain the adjacency matrix without self-connection
    adj = adj - ss.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
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
        adj = ss.coo_matrix(adj)
        adj.eliminate_zeros()

    print("The construction of Laplacian matrix is finished!")
    print("The time cost of construction: ", time.time() - start_time)
    adj = ss.coo_matrix(adj)
    return adj


def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    # adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    # lp = ss.eye(adj.shape[0]) - adj_wave
    return adj_wave


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape


def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {} ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

if __name__ == '__main__':
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    for args.knns in [5,10,15,20,25,30,35,40]:
        feature_list, adj_list, labels, idx_labeled, idx_unlabeled,Homogeneity = load_data(args, device)
        results_direction = './results/Homogeneity/' + args.dataset + '_Homogeneity_results.txt'
        fp = open(results_direction, "a+", encoding="utf-8")
        fp.write("\ndataset_name: {}\n".format(args.dataset))
        fp.write("knn: {}  |  ".format(args.knns))
        fp.write("Homogeneity: {:.2f}  |  ".format(Homogeneity*100))
        fp.close()