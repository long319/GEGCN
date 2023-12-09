"""
@Project   : MvGCN
@Time      : 2021/10/4
@Author    : Zhihao Wu
@File      : utils.py
"""
import torch
from texttable import Texttable
from sklearn import metrics

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')

    return ACC, P, R, F1


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    # print(R.shape[0], K1.shape[0], K2.shape[0])
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC

def compute_renormalized_adj(adj, device):
    # adj = adj + torch.mul(adj.t(), (adj.t() > adj)) - torch.mul(adj, (adj.t() > adj))
    adj_ = torch.eye(adj.shape[0]).to(device) + adj
    rowsum = torch.tensor(adj_.sum(1)).to(device)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5)).to(device)  # degree matrix
    adj_hat = (degree_mat_inv_sqrt).mm(adj_).mm(degree_mat_inv_sqrt)
    return adj_hat