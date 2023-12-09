"""
@Project   : MvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, compute_renormalized_adj
from Dataloader import load_data, construct_laplacian,load_data_Isogram
import tqdm
import random
import scipy.sparse as ss
import warnings
from itertools import product
import scipy.sparse as sp
import time
warnings.filterwarnings("ignore")
from tqdm import tqdm
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

class Linerlayer(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(Linerlayer, self).__init__()
        self.weight = glorot_init(inputdim, outputdim)
        # self.device = device
    def forward(self, x, sparse=False):
        if sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        return x


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, device, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.device = device
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.sparse.mm(adj, x)
        if self.activation==None:
            return x
        else:
            return self.activation(x)


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class Evaluator(nn.Module):
    def __init__(self, in_size):
        super(Evaluator, self).__init__()
        self.w = nn.Parameter(torch.rand_like(torch.eye(in_size)), requires_grad=True)
    def forward(self,adj):
        score = torch.sparse.mm(adj, self.w)
        score = F.sigmoid(score)
        return score


class GCN(nn.Module):
    def __init__(self,input_dim, output_dim,device,args):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim=output_dim
        self.device =device
        # self.A=A
        self.hidden_dim = input_dim // 2
        self.gc1 = GraphConvSparse(input_dim, 64, self.device)
        self.gc2 = GraphConvSparse(64, output_dim, self.device)
        self.theta = nn.Parameter(torch.FloatTensor([-1.5]), requires_grad=True)

    def forward(self,adj,X):
        theta1, theta2 = self.thred_proj(self.theta)
        w1 = theta2 / (theta2 - theta1)
        w2 = theta1 / (theta2 - theta1)
        adj = w1 * F.relu(- theta1 + adj) - w2 * F.relu(- theta2 + adj)
        adj = compute_renormalized_adj(adj, self.device)
        g = F.relu(self.gc1(X, adj))
        g = F.dropout(g, 0.3)
        g = self.gc2(g, adj)

        return g , adj
    def thred_proj(self, theta):
        theta_sigmoid = torch.sigmoid(theta)
        theta1 = theta_sigmoid / 2
        theta2 = theta_sigmoid / 2 + 0.1
        # theta1 = torch.broadcast_to(theta1,(N,N))
        # theta2 = torch.broadcast_to(theta2,(N,N))

        return theta1, theta2

    def learnedfunction(self,x,theta1,theta2):
        if x < theta1:
            return 0
        elif x < theta2:
            return (x - theta1) / (theta2 - theta1) * theta2
        else:
            return x

class Generator(nn.Module):
    def __init__(self, nfeat, nnodes, edgeidx, nhid=32, nlayers=2, device=None, args=None):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))
        self.row, self.col = edgeidx
        self.edge_index = edgeidx
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        # self.nnodes = nnodes

    def forward(self, x, inference=False):
        edge_index = self.edge_index
        edge_embed = (torch.cat([x[edge_index[0]],
                x[edge_index[1]]], axis=1)+torch.cat([x[edge_index[1]],
                x[edge_index[0]]], axis=1))/2
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)
        return edge_embed

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

class Decomposition(nn.Module):
    def __init__(self, inputdim_list, outputdim):
        super(Decomposition, self).__init__()
        self.W = nn.ModuleList()
        for i in range(len(inputdim_list)):
            self.W.append(Linerlayer(inputdim_list[i],outputdim))
    def forward(self, feature_list):
        de_feature_list = []
        for i in range(len(feature_list)):
            x = self.W[i](feature_list[i],sparse=True)
            de_feature_list.append(x)
        return de_feature_list

class DeepMvNMF(nn.Module):
    def __init__(self, input_dims, en_hidden_dims, num_views, device):
        super(DeepMvNMF, self).__init__()
        self.encoder = nn.ModuleList()
        self.mv_decoder = nn.ModuleList()
        self.device = device
        # self.decrease = nn.Linear(en_hidden_dims[i], en_hidden_dims[i + 1])
        for i in range(len(en_hidden_dims)-1):
            # self.encoder.append(nn.Linear(en_hidden_dims[i], en_hidden_dims[i+1]))
            self.encoder.append(Linerlayer(en_hidden_dims[i], en_hidden_dims[i+1]))
        for i in range(num_views):
            decoder = nn.ModuleList()
            de_hidden_dims = [input_dims[i]]
            for k in range(1, len(en_hidden_dims)):
                de_hidden_dims.insert(0, en_hidden_dims[k])
            # print(de_hidden_dims)
            for j in range(len(de_hidden_dims)-1):
                decoder.append(nn.Linear(de_hidden_dims[j], de_hidden_dims[j+1]))
            self.mv_decoder.append(decoder)
        # print(self.encoder)
        # print(self.mv_decoder)

    def forward(self, input):
        z = input
        for layer in self.encoder:
            z = F.relu(layer(z,sparse=True))
        x_hat_list = []
        for de in self.mv_decoder:
            x_hat = z
            for layer in de:
                x_hat = F.relu(layer(x_hat))
            x_hat_list.append(x_hat)
        return z, x_hat_list

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    # args.path = 'D:\\code\\datasets\\'
    args.device = device
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    feature_list, adj_list, labels, idx_labeled, idx_unlabeled = load_data(args, device)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    d = feature_list[0].shape[1]

    edge_index_temp = sp.coo_matrix(adj_list[0].cpu())
    # edge_index_temp = adj_splist[0]
    values = edge_index_temp.data  # 边上对应权重值weight
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式
    row , col = edge_index_A
    num_view = len(feature_list)
    input_dims = []
    edge_index_A = edge_index_A.to(device)
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])
    pge = Generator(nfeat=d, nnodes = N, edgeidx=edge_index_A ,device=device, args=args).to(device)
    adj = Generator.inference(feature_list[0])
    # adj_syn = sp.coo_matrix((emb,(row,col)),shape=(N , N))
    print(adj.to_dense().squeeze())