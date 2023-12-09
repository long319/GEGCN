import warnings
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, compute_renormalized_adj
from Dataloader import load_data, construct_adjacency_matrix
from torch.autograd import Variable
from model import Generator, Evaluator, GCN,DeepMvNMF,Decomposition
import scipy.sparse as ss
from plotheatmap import heapMapPlot,draw_plt
import scipy.sparse as sp
import scipy.io as sio

def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)

def permute_adj(affinity, labels, n_class):
    new_ind = []
    for i in range(n_class):
        ind = np.where(labels == i)[0].tolist()
        new_ind += ind
    return affinity[new_ind, :][:, new_ind]

def train(args, device):
    feature_list, adj_list, labels, idx_labeled, idx_unlabeled = load_data(args, device)
    mask = torch.zeros_like(adj_list[0]).bool().to(device)
    for adj in adj_list:
        mask = mask | adj.bool()
    mask = torch.where(mask, 1, 0)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)
    # for i in range(len(adj_list)):
    #     adj_temp = permute_adj(adj_list[i].detach().cpu(), labels.detach().cpu(), num_classes)
    #     heapMapPlot(adj_temp.detach().cpu() + np.eye(N),'before',args.dataset+'_view'+str(i))
    # adj_temp  = adj_list[0]
    # for i in range(1,len(adj_list)):
    #     adj_temp = adj_list[i] + adj_temp
    # adj_temp = adj_temp / len(adj_list)
    # adj_temp = permute_adj(adj_temp.detach().cpu(), labels.detach().cpu(), num_classes)
    # heapMapPlot(adj_temp.detach().cpu() + np.eye(N), 'before', args.dataset + '_viewfusion')

    input_dims = []
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])
    #维度太大时选择降维
    if args.dataset == 'Reuters':
        Defeature = Decomposition(input_dims, 256).to(device)
        x_de = Defeature(feature_list)
        input_dims = []
        for i in range(num_view): # multiview data { data includes features and ... }
            input_dims.append(x_de[i].shape[1])
            feature_list[i] = x_de[i].detach()
        torch.cuda.empty_cache()

##########################################################################
    en_hidden_dims = [N, 128]
    DMF_model = DeepMvNMF(input_dims, en_hidden_dims, num_view, device).to(device)
    optimizer_DMF = torch.optim.Adam(DMF_model.parameters(), lr=1e-3, weight_decay=5e-5)
    identity = torch.eye(feature_list[0].shape[0]).to(device)

    with tqdm(total=1000, desc="Pretraining") as pbar:
        for epoch in range(1000):
            shared_z, x_hat_list = DMF_model(identity)
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            optimizer_DMF.zero_grad()
            loss_DMF.backward()
            optimizer_DMF.step()
            pbar.set_postfix({'Loss': '{:.6f}'.format(loss_DMF.item())})
            pbar.update(1)

    shared_z = shared_z.detach()

    edge_index_A = sp.coo_matrix(mask.cpu())
    indices = np.vstack((edge_index_A.row, edge_index_A.col))  # 我们真正需要的coo形式
    edge_index_A = torch.LongTensor(indices).to(device)

    # print(edge_index_A)
    #MODEL
    G = Generator(nfeat=shared_z.shape[1], nnodes = N, edgeidx=edge_index_A ,device=device, args=args).to(device)
    D = Evaluator(N).to(device)
    GCN_thred = GCN(shared_z.shape[1],num_classes,args.device,args).to(device)
    #optimizer
    optimizer_Generator = torch.optim.Adam(G.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer_Evaluator = torch.optim.Adam(D.parameters(), lr=1e-2, weight_decay=5e-4)
    optimizer_GCN_thred = torch.optim.Adam(GCN_thred.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_function1 = torch.nn.NLLLoss()
    loss_function2 = torch.nn.BCELoss()

    Best_Acc=0
    Best_F1=0
    Loss_list_G = []
    Loss_list_D = []
    Loss_list = []
    ACC_list = []
    F1_list = []
    begin_time = time.time()

    real_ = torch.Tensor(N, 1).fill_(1.0).to(device)
    fake_ = torch.Tensor(N, 1).fill_(0.0).to(device)
    with tqdm(total=args.num_epoch, desc="training", position=0) as pbar:
        for epoch in range(args.num_epoch):
            #shared representation about downstream task
            shared_z, x_hat_list = DMF_model(identity)
            generated_A = G.forward(shared_z)
            generated_A = torch.sparse_coo_tensor(edge_index_A, generated_A.flatten(), requires_grad=False)
            adj = generated_A.to_dense()
            result, adj_p = GCN_thred(adj, shared_z)
            output = F.log_softmax(result, dim=1)
            loss_ce = loss_function1(output[idx_labeled], labels[idx_labeled])
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            loss_share = loss_ce + loss_DMF
            optimizer_DMF.zero_grad()
            loss_share.backward()
            optimizer_DMF.step()

            shared_z = shared_z.detach()
            #encoder
            optimizer_Generator.zero_grad()
            generated_A = G.forward(shared_z)
            generated_A = F.sigmoid(generated_A)
            generated_A = torch.sparse_coo_tensor(edge_index_A, generated_A.flatten(), requires_grad=True)
            adj = generated_A.to_dense()
            result, adj_p = GCN_thred(adj, shared_z)
            output = F.log_softmax(result, dim=1)
            loss_ce = loss_function1(output[idx_labeled], labels[idx_labeled])

            dist = D(adj)
            fake = torch.diag(dist).reshape(-1,1)
            # fake = dist.reshape(-1,1)
            loss_g = loss_function2(fake, real_)
            loss_togeter = loss_g + loss_ce
            loss_togeter.backward()
            optimizer_Generator.step()

            #discriminator
            optimizer_Evaluator.zero_grad()
            generated_A = G.forward(shared_z)
            generated_A = F.sigmoid(generated_A)
            generated_A = torch.sparse_coo_tensor(edge_index_A, generated_A.flatten(), requires_grad=False)
            dist = D(generated_A)
            fake = torch.diag(dist).reshape(-1, 1)
            # fake = dist.reshape(-1, 1)

            real_d = []
            for i in range(num_view):
                dist = D(adj_list[i])
                real_d.append(torch.diag(dist).reshape(-1, 1))
                # real_d.append(dist.reshape(-1, 1))
            loss_e =(sum([loss_function2(real_d[i], real_)/num_view for i in range(num_view)])+loss_function2(fake,fake_))/2
            loss_e.backward()
            optimizer_Evaluator.step()

            #GCN
            GCN_thred.train()
            adj = generated_A.to_dense()
            adj = adj.detach()
            result, adj_p = GCN_thred(adj, shared_z)
            output = F.log_softmax(result, dim=1)
            loss_ce = loss_function1(output[idx_labeled], labels[idx_labeled])

            optimizer_GCN_thred.zero_grad()
            loss_ce.backward()
            optimizer_GCN_thred.step()
            with torch.no_grad():
                GCN_thred.eval()
                # output, _, _ = model(feature_list, lp_list, args.Lambda, args.ortho)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, _, _, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
                if ACC > Best_Acc:
                    Best_Acc = ACC
                    Best_F1 = F1
                    # bestre = result
                pbar.set_postfix({'Loss_G': '{:.6f}'.format(loss_g.item()),'Loss_d': '{:.6f}'.format(loss_e.item()),'Loss_ce': '{:.6f}'.format(loss_ce.item()),
                                  'ACC': '{:.2f}'.format(ACC * 100),'Best acc': '{:.4f}'.format(Best_Acc*100),'Best F1': '{:.4f}'.format(Best_F1*100)})
                # pbar.set_postfix({'Loss_ce': '{:.6f}'.format(loss_ce.item()),
                #                   'ACC': '{:.2f}'.format(ACC * 100),'Best acc': '{:.4f}'.format(Best_Acc*100),'Best F1': '{:.4f}'.format(Best_F1*100)})
                pbar.update(1)
                Loss_list.append(float(loss_ce.item()))
                Loss_list_G.append(float(loss_g.item()))
                Loss_list_D.append(float(loss_e.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)
    # heapMapPlot(input_d.detach().cpu(),'input')
    # adj_plot = permute_adj(np.array(adj.detach().cpu()), labels.detach().cpu(), num_classes)
    # adj_p_plot = permute_adj(np.array(adj_p.detach().cpu()), labels.detach().cpu(), num_classes)
    # heapMapPlot(adj_plot + np.eye(N),'before',args.dataset+'_before')
    # heapMapPlot(adj_p_plot,'after',args.dataset+'_after')
    # draw_plt(args.dataset, bestre.detach().cpu().numpy(), labels.detach().cpu().numpy())

    cost_time = time.time() - begin_time
    GCN_thred.eval()
    z, adj_p = GCN_thred(adj, shared_z)
    print("Evaluating the model")
    pred_labels = torch.argmax(z, 1).cpu().detach().numpy()
    ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("------------------------")
    print("ratio = ",args.ratio)
    print("ACC:   {:.2f}".format(ACC * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("------------------------")

    return ACC, P, R, F1, cost_time, Loss_list,Loss_list_G,Loss_list_D, ACC_list, F1_list,Best_Acc


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    save_direction='./adj_matrix/' + args.dataset + '/'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    for ratio in [0.1]:
        all_ACC = []
        all_P = []
        all_R = []
        all_F1 = []
        all_TIME = []
        args.ratio = ratio
        for i in range(args.n_repeated):
            torch.cuda.empty_cache()
            ACC, P, R, F1, Time, Loss_list,Loss_list_G,Loss_list_D, ACC_list, F1_list ,Best_Acc= train(args, device)
            all_ACC.append(ACC)
            all_P.append(P)
            all_R.append(R)
            all_F1.append(F1)
            all_TIME.append(Time)

            print("-----------------------")
            print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
            print("P  : {:.2f} ({:.2f})".format(np.mean(all_P) * 100, np.std(all_P) * 100))
            print("R  : {:.2f} ({:.2f})".format(np.mean(all_R) * 100, np.std(all_R) * 100))
            print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
            print("-----------------------")
            # if args.save_results:
            #     experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            #     results_direction = './results/' + args.dataset + '_results.txt'
            #     fp = open(results_direction, "a+", encoding="utf-8")
            #     # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
            #     fp.write(format(experiment_time))
            #     fp.write("\ndataset_name: {}\n".format(args.dataset))
            #     fp.write("knn: {}  |  ".format(args.knns))
            #     fp.write("alpha: {}  |  ".format(args.alpha))
            #     fp.write("beta: {}  |  ".format(args.beta))
            #     fp.write("theta: {}  |  ".format(args.theta))
            #     fp.write("w: {}  |  ".format(args.w))
            #     fp.write("ratio: {}  |  ".format(args.ratio))
            #     fp.write("epochs: {}  |  ".format(args.num_epoch))
            #     fp.write("lr: {}  |  ".format(args.lr))
            #     fp.write("wd: {}\n".format(args.weight_decay))
            #     # fp.write("lambda: {}  |  ".format(args.Lambda))
            #     # fp.write("alpha: {}\n".format(args.alpha))
            #     # fp.write("layer: {}\n".format(str_layers))
            #     fp.write("ACC:  {:.2f} ({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
            #     fp.write("F1 :  {:.2f} ({:.2f})\n".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
            #     fp.write("Best ACC:  {:.2f}\n".format(Best_Acc*100))
            #     fp.write("Time:  {:.2f} ({:.2f})\n\n".format(np.mean(all_TIME), np.std(all_TIME)))
            #     fp.close()
