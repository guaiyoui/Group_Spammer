import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import early_stopping, remove_nodes_from_walks, sgc_precompute, \
    get_classes_statistic
from models import get_model
from metrics import accuracy, f1, f1_isr
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sampling_methods import *
import os
import datetime
import json
import pandas as pd
from community import community_louvain
from torch_geometric.utils import get_laplacian
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
import matplotlib.pyplot as plt
import psgd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Arguments
args = get_citation_args()
def plot(index, data, figure_name):

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(index, data)

    # 找出最小值、最大值和最后的值
    min_value = min(data)
    max_value = max(data)
    last_value = data[-1]
    min_index = data.index(min_value)
    max_index = data.index(max_value)
    last_index = len(data) - 1

    # 标注最小值（左上角）
    plt.annotate(f'Min: {min_value}', 
                 xy=(min_index, min_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction')

    # 标注最大值（中间上方）
    plt.annotate(f'Max: {max_value}', 
                 xy=(max_index, max_value), 
                 xytext=(0.5, 0.95), 
                 textcoords='axes fraction',
                 ha='center')

    # 标注最后的值（右上角）
    plt.annotate(f'Last: {last_value}', 
                 xy=(last_index, last_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right')
    
    # 设置标题和轴标签
    plt.title(figure_name)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 保存图表
    plt.savefig("./figures/"+figure_name+".png")
    plt.close()

def loss_function_laplacian_regularization(output, train_labels, edge_index):
    loss_cls = F.cross_entropy(output, train_labels)
    
    # 计算稀疏格式的图拉普拉斯矩阵
    lap_sp = get_laplacian(edge_index, normalization='sym')[0]
    lap_sp = sp.FloatTensor(lap_sp)
    
    # 使用稀疏矩阵乘法计算损失
    loss_lap = torch.sum((lap_sp @ output.T) ** 2)
    
    return loss_cls + 0.1 * loss_lap

def loss_function_consistency_regularization(model, x, edge_index, train_labels, selected_nodes):
    # 计算原始输入的预测结果
    output = model(x, edge_index)
    output_select = output[selected_nodes, :]
    loss_cls = F.cross_entropy(output_select, train_labels)
    
    # Generate adversarial samples by adding random noise
    adv_x = x + 0.1 * torch.randn_like(x)

    # Compute the adversarial output
    adv_output = model(adv_x, edge_index)

    # Compute the consistency loss
    loss_cons = F.kl_div(adv_output, output.detach(), reduction='batchmean')

    return loss_cls + 0.1 * loss_cons

def loss_function_subgraph_regularization(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    # loss_cls = F.cross_entropy(output_selected, train_labels)
    # loss_cls = F.nll_loss(output_selected, train_labels)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_cls = loss_fn(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()
    # node_membership = node_membership.T
    
    # 计算同一子图内节点预测差异的平方和
    loss_subgraph = torch.sum(torch.matmul(node_membership.T, torch.matmul(adj, node_membership)))
    
    return loss_cls + args.weight_loss_subgraph * loss_subgraph

def loss_function_subgraph_regularization_v1(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()

    # 计算邻域内节点预测相似性
    # 计算邻域内节点预测的相似度矩阵
    sim_matrix = torch.matmul(output, output.T)  # 计算预测值的相似度矩阵
    neighborhood_sim = torch.sum(sim_matrix * adj)  # 在邻接矩阵中加权相似度矩阵
    
    # 最小化相似性
    loss_similarity = torch.mean(neighborhood_sim)
    
    return loss_cls + 0.1 * loss_similarity

def loss_function_local_consistency(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    # 分类损失
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    # 计算邻接矩阵
    adj = edge_index.coalesce()
    adj = adj.to_dense()
    
    # 获取节点嵌入
    node_membership = model.get_node_embedding()
    
    # 计算每个节点及其邻域的预测差异
    loss_local_consistency = 0
    for node in selected_nodes:
        neighbors = torch.nonzero(adj[node, :]).squeeze()  # 获取邻域节点
        if len(neighbors) > 0:
            node_output = output[node, :]
            neighbors_output = output[neighbors, :]
            # 计算预测差异
            loss_local_consistency += torch.sum((node_output - neighbors_output) ** 2)
    
    return loss_cls + 0.1 * loss_local_consistency


def train_GCN(model, adj, selected_nodes, val_nodes,
             features, train_labels, val_labels,
             epochs=args.epochs, weight_decay=args.weight_decay,
             lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    
    t = perf_counter()
    best_acc_val = 0
    should_stop = False
    stopping_step = 0

    patience = 150

    for epoch in range(epochs):
        
        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index=adj)
        output = output[selected_nodes, :]
        # print(f'output.size(): {output.size()}')

        # loss_train = F.cross_entropy(output, train_labels)
        # loss_train = F.nll_loss(output, train_labels)
        # loss_train = loss_function_laplacian_regularization(output, train_labels, adj)
        # loss_train = loss_function_consistency_regularization(model, features, adj, train_labels, selected_nodes)
        loss_train = loss_function_subgraph_regularization(model, features, adj, train_labels, selected_nodes)
        # loss_train = loss_function_local_consistency(model, features, adj, train_labels, selected_nodes)

        # loss_train.backward()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        # 计算验证集准确率
        with torch.no_grad():
            model.eval()
            val_output = model(features, edge_index=adj)
            val_output = val_output[val_nodes, :]
            # print(val_output)
            # val_acc = accuracy(val_output, val_labels)
            val_acc, recall_val, precision_val = f1_isr(val_output, val_labels)
        
        # 早停逻辑
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            stopping_step = 0  # 重新计数
        else:
            stopping_step += 1
            if stopping_step >= patience:
                should_stop = True
            

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        output = output[val_nodes, :]
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        # print('macro_val: {}'.format(macro_val))
        # print(output)
        f1_val, recall_val, precision_val = f1_isr(output, val_labels)
        # print('f1_val_isr: {}'.format(f1_val))
    return model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val

def test_GCN(model, adj, features, test_mask, test_labels, all_test_idx, all_test_labels, save_name=None, dataset_name=None, sample_global=False):
    model.eval()
    output_all = model(features, adj)

    output_test_all = output_all[all_test_idx, :]
    output_test_all_preds = output_test_all.max(1)[1]
    if sample_global:
        path = "../detection_results/"+dataset_name+"/our_"+save_name+"_all_sample_global.txt"
        metric_path_all = "../detection_results/"+dataset_name+"/our_"+save_name+"_all_sample_global_metics.txt"
    else:
        path = "../detection_results/"+dataset_name+"/our_"+save_name+"_all.txt"
        metric_path_all = "../detection_results/"+dataset_name+"/our_"+save_name+"_all_metrics.txt"

    with open(path, 'w') as file:
        file.write("sample_index\tprediction\n")
        for i, pred in zip(all_test_idx, output_test_all_preds):
            file.write(f'{i+1}\t{pred}\n')

    output_in_test = output_all[test_mask, :]
    output_in_test_preds = output_in_test.max(1)[1]
    if sample_global:
        path = "../detection_results/"+dataset_name+"/our_"+save_name+"_sample_global.txt"
        metric_path = "../detection_results/"+dataset_name+"/our_"+save_name+"_sample_global_metics.txt"
    else:
        path = "../detection_results/"+dataset_name+"/our_"+save_name+".txt"
        metric_path = "../detection_results/"+dataset_name+"/our_"+save_name+"_metrics.txt"
    
    with open(path, 'w') as file:
        file.write("sample_index\tprediction\n")
        for i, pred in zip(test_mask, output_in_test_preds):
            file.write(f'{i+1}\t{pred}\n')
    
    micro_test_all, macro_test_all = f1(output_test_all, all_test_labels)
    f1_test_all, recall_test_all, precision_test_all = f1_isr(output_test_all, all_test_labels)

    with open(metric_path_all, 'w') as f_out:
        f_out.write(f"F-measure: {f1_test_all:.4f}\nRecall: {recall_test_all:.4f}\nPrecision: {precision_test_all:.4f}\n")
    # print(f"Saved evaluation metrics to {metric_path_all}")


    micro_test, macro_test = f1(output_in_test, test_labels)
    f1_test, recall_test, precision_test = f1_isr(output_in_test, test_labels)
    with open(metric_path, 'w') as f_out:
        f_out.write(f"F-measure: {f1_test:.4f}\nRecall: {recall_test:.4f}\nPrecision: {precision_test:.4f}\n")
    # print(f"Saved evaluation metrics to {metric_path}")


    # print(f'macro_test_all: {macro_test_all}, f1_test_all: {f1_test_all}, macro_test: {macro_test}, f1_test: {f1_test}')

    return macro_test_all, f1_test_all, macro_test, f1_test


def ensure_nonrepeat(idx_train, selected_nodes):
    for node in idx_train:
        if node in selected_nodes:
            raise Exception(
                'In this iteration, the node {} need to be labelled is already in selected_nodes'.format(node))
    return

def augment_feature(feature, nx_G):
    print("===== 1. The modularity-based feature augmentation. =====")
    partition = community_louvain.best_partition(nx_G)
    modularity = community_louvain.modularity(partition, nx_G)
    print(f"the modularity of community is {modularity}")
    # 创建一个字典存储每个社区的modularity值
    node_modularity = {}
    for community in set(partition.values()):
            # 取出该社区的节点
        nodes_in_community = [node for node, comm in partition.items() if comm == community]
        # 计算该社区在整体中的modularity贡献
        subgraph = nx_G.subgraph(nodes_in_community)
        # print(subgraph)
        community_partition = {node: community for node in nodes_in_community}
        community_modularity = community_louvain.modularity({**partition, **community_partition}, nx_G)
        # 分配给该社区中的每个节点
        for node in nodes_in_community:
            node_modularity[node] = community_modularity
    
    augmented_mod_feat = []
    for i in range(feature.shape[0]):
        if i in node_modularity:
            augmented_mod_feat.append(node_modularity[i])
        else:
            augmented_mod_feat.append(0)
    # kcore based 

    augmented_core_feat = []
    print("===== 2. The k-core-based feature augmentation. =====")
    # Calculate k-core values for each node
    # Remove self-loops before calculating core numbers
    G_no_selfloops = nx_G.copy()
    G_no_selfloops.remove_edges_from(nx.selfloop_edges(G_no_selfloops))
    core_numbers = nx.core_number(G_no_selfloops)
    for i in range(feature.shape[0]):
        if i in core_numbers:
            augmented_core_feat.append(core_numbers[i])
        else:
            augmented_core_feat.append(0)
    
    # print(augmented_core_feat)
    result = np.column_stack((feature, np.array(augmented_mod_feat), np.array(augmented_core_feat)))

    return result


class run_wrapper():
    def __init__(self, dataset, normalization, cuda):
        if dataset in ['spammer', 'amazon', 'yelp', 'he_amazon']:

            self.graph = None
            # graph_data = np.loadtxt("../Unsupervised_Spammer_Learning/data_graph/spammer_edge_index.txt", delimiter=' ', dtype=int)
            print("start loading J01Network")
            graph_data = np.loadtxt(args.data_path+"J01Network.txt", delimiter=' ', dtype=int)
            graph_data[:,0] = graph_data[:,0] - 1
            graph_data[:,1] = graph_data[:,1] - 1
            self.nx_G = nx.Graph()
            self.nx_G.add_edges_from(graph_data)
            
            # 获取最大的节点编号
            max_node = max(graph_data.max(), graph_data.max())  # 确保编号从0到max_node

            # 手动添加孤立节点，确保包含所有节点
            for node in range(max_node + 1):
                if node not in self.nx_G:
                    self.nx_G.add_node(node)

            self.graph = self.nx_G

            print("start constructing adj")
            edge_tensor = torch.from_numpy(graph_data).long()
            indices = edge_tensor.t().contiguous()
            num_edges = edge_tensor.shape[0]
            values = torch.ones(num_edges)
            num_nodes = edge_tensor.max().item() + 1
            adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
            adj = adj.coalesce()
            # adj = adj.to('cuda:0')
            adj = adj.cuda()
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            row_sum[row_sum == 0] = 1  # 避免除以零
            values_normalized = 1.0 / row_sum[adj.indices()[0]]
            adj_normalized = torch.sparse_coo_tensor(adj.indices(), values_normalized, adj.size())
            self.adj = adj_normalized
            # self.adj = adj
            print(self.adj)

            print("start loading features")
            
            features = np.loadtxt(args.data_path+"UserFeature.txt", delimiter='\t')
            features = augment_feature(features, self.nx_G)
            self.features = torch.from_numpy(features).float().cuda()

            print("start loading labels")
            labels_data = pd.read_csv(args.data_path+"UserLabel.txt", sep=' ', usecols=[1, 2])
            labels_data = labels_data.to_numpy()
            self.labels = torch.from_numpy(labels_data[:, 1]).cuda()
            
            training_data = np.loadtxt(args.data_path+"Training_Testing/"+args.test_percents+"/train_4.csv", delimiter=' ', dtype=int)
            testing_data = np.loadtxt(args.data_path+"Training_Testing/"+args.test_percents+"/test_4.csv", delimiter=' ', dtype=int)
    
            self.idx_test = torch.from_numpy(testing_data[:,0] - 1).cuda()

            self.idx_non_test = (training_data[:,0]-1).tolist() 

            self.idx_test_ori = torch.from_numpy(testing_data[:,0] - 1).cuda()

        self.dataset = dataset
        # print(f'self.labels: {self.labels, self.labels.shape}')
        # print(f'self.adj: {self.adj}')
        # print(f'self.feature: {self.features, self.features.shape}')
        # print(f'self.idx_test is {len(self.idx_test)}, self.idx_non_test is {len(self.idx_non_test)}')
        # print('finished loading dataset')
        self.raw_features = self.features
        if args.model == "SGC":
            self.features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
            print("{:.4f}s".format(precompute_time))
            if args.strategy == 'featprop':
                self.dis_features = self.features
        else:
            if args.strategy == 'featprop':
                self.dis_features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
                # torch.save(self.dis_features.data, 'visualization/featprop_feat.pt')
                # input('wait')


    def run(self, strategy, num_labeled_list=[10, 15, 20, 25, 30, 35, 40, 50], max_budget=160, seed=1):
        # set_seed(seed, args.cuda)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if cuda: torch.cuda.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
            
            # 为了确保在使用 cudnn 的情况下结果是可复现的
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        max_budget = num_labeled_list[-1]
        if strategy in ['ppr', 'pagerank', 'pr_ppr', 'mixed', 'mixed_random', 'unified']:
            # print('strategy is ppr or pagerank')
            # nx_G = nx.from_dict_of_lists(self.graph)
            nx_G = self.nx_G
            PR_scores = nx.pagerank(nx_G, alpha=0.85)
            # print('PR_scores: ', PR_scores.keys())
            # print(len(PR_scores.keys()))
            # print(nx_G.number_of_nodes())
            # print('PR_scores: ', PR_scores)
            nx_nodes = nx.nodes(nx_G)
            original_weights = {}
            for node in nx_nodes:
                original_weights[node] = 0.

        idx_non_test = self.idx_non_test.copy()
        # print('len(idx_non_test) is {}'.format(len(idx_non_test)))
        # Select validation nodes.
        # num_val = 500
        num_val = 10
        idx_val = np.random.choice(idx_non_test, num_val, replace=False)
        idx_non_test = list(set(idx_non_test) - set(idx_val))

        # initially select some nodes.
        L = 5
        selected_nodes = np.random.choice(idx_non_test, L, replace=False)
        idx_non_test = list(set(idx_non_test) - set(selected_nodes))

        model = get_model(model_opt=args.model, nfeat=self.features.size(1), nsample=self.features.size(0),nclass=2, nhid=args.hidden, dropout=args.dropout,
                          cuda=args.cuda)
        # model.reset_parameters()
        budget = 20
        steps = 6
        pool = idx_non_test
        # print('len(idx_non_test): {}'.format(len(idx_non_test)))
        np.random.seed() # cancel the fixed seed
        if args.sample_global:
            all_test_idx = list(set(self.idx_test).union(set(pool)))
            pool = list(set(self.idx_test.cpu().numpy().tolist()).union(set(pool)))
            test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))
        else:
            all_test_idx = list(set(self.idx_test).union(set(pool)))
            test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))

        if args.model == 'GCN_update':
            args.lr = 0.01
            model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                                self.labels[selected_nodes],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)
        # print('-------------initial results------------')
        # print('micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
        # Active learning
        # print('strategy: ', strategy)
        cur_num = 0
        val_results = {'acc': [], 'micro': [], 'macro': [], 'f1': [], "recall":[], "precision":[]}
        test_results = {'macro_test_all': [], 'f1_test_all': [], 'macro_test': [], 'f1_test': []}

        uncertainty_results = {}
        if strategy == 'rw':
            self.walks = remove_nodes_from_walks(self.walks, selected_nodes)
        if strategy == 'unified':
            nodes = nx.nodes(nx_G)
            uncertainty_score = get_uncertainty_score(model, self.features, nodes)
            init_weights = {n: float(uncertainty_score[n]) for n in nodes}
            for node in selected_nodes:
                init_weights[node] = 0
            uncertainty_results[5] = {'selected_nodes': selected_nodes.tolist(), 'uncertainty_scores': init_weights}


        time_AL = 0
        for i in range(len(num_labeled_list)):
            if num_labeled_list[i] > max_budget:
                break
            budget = num_labeled_list[i] - cur_num
            cur_num = num_labeled_list[i]
            t1 = perf_counter()
            if strategy == 'random':
                idx_train = query_random(budget, pool)
            elif strategy == 'uncertainty':
                if args.model == 'GCN_update':
                    idx_train = query_uncertainty_GCN(model, self.adj, self.features, budget, pool)
                else:
                    idx_train = query_uncertainty(model, self.features, budget, pool)
            elif strategy == 'largest_degrees':
                if args.dataset not in ['cora', 'citeseer', 'pubmed']:
                    idx_train = query_largest_degree(self.graph, budget, pool)
                else:
                    idx_train = query_largest_degree(nx.from_dict_of_lists(self.graph), budget, pool)
            elif strategy == 'coreset_greedy':
                idx_train = qeury_coreset_greedy(self.features, list(selected_nodes), budget, pool)
            elif strategy == 'featprop':
                idx_train = query_featprop(self.dis_features, budget, pool)
            elif strategy == 'pagerank':
                idx_train = query_pr(PR_scores, budget, pool)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))

            time_AL += perf_counter() - t1
            assert len(idx_train) == budget
            ensure_nonrepeat(idx_train, selected_nodes)
            selected_nodes = np.append(selected_nodes, idx_train)
            pool = list(set(pool) - set(idx_train))

            if args.sample_global:
                # print("============sample global=======")
                all_test_idx = list(set(pool))
                test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()).intersection(set(pool)))
                # print(len(test_idx_in_test))
                # print(len(all_test_idx))
            else:
                # print("============sample only in training=======")
                test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))
                all_test_idx = list(set(pool).union(test_idx_in_test))
                # print(len(test_idx_in_test))
                # print(len(all_test_idx))
            

            if args.model == 'GCN_update':
                model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                             self.labels[selected_nodes],
                                                                             self.labels[idx_val],
                                                                             args.epochs, args.weight_decay, args.lr,
                                                                             args.dropout)
            print(f"the number of labels is {num_labeled_list[i]}")
            if args.model == 'GCN_update':
                macro_test_all, f1_test_all, macro_test, f1_test = test_GCN(model, self.adj, self.features, test_idx_in_test, self.labels[test_idx_in_test], all_test_idx, self.labels[all_test_idx], save_name=args.test_percents, dataset_name=args.dataset, sample_global=args.sample_global)
            

            # print('f1_val_isr: {}'.format(f1_val))
            print('f1_test_isr: {}'.format(f1_test))

            macro_test = round(macro_test, 4)
            f1_test = round(f1_test, 4)
            macro_test_all = round(macro_test_all, 4)
            f1_test_all = round(f1_test_all, 4)

            test_results['macro_test_all'].append(macro_test_all)
            test_results['f1_test_all'].append(f1_test_all)
            test_results['macro_test'].append(macro_test)
            test_results['f1_test'].append(f1_test)

        # print('AL Time: {}s'.format(time_AL))
        return val_results, test_results, get_classes_statistic(self.labels[selected_nodes].cpu().numpy()), time_AL



if __name__ == '__main__':

    if args.dataset == 'spammer':
        num_labeled_list = [i for i in range(10,151,10)]
    elif args.dataset == 'amazon':
        if args.test_percents in ['50percent', '30percent', '10percent']:
            num_labeled_list = [i for i in range(10,721,10)]
        else:
            num_labeled_list = [i for i in range(10,401,10)]
    elif args.dataset == 'yelp':
        num_labeled_list = [10, 20, 30, 40] + [i for i in range(50,1001,50)]
    elif args.dataset == 'he_amazon':
        if args.test_percents in ['50percent', '30percent', '10percent']:
            num_labeled_list = [i for i in range(10,341,10)]
        else:
            num_labeled_list = [i for i in range(10,171,10)]
    num_interval = len(num_labeled_list)

    val_results = {'micro': [[] for _ in range(num_interval)],
                   'macro': [[] for _ in range(num_interval)],
                   'acc': [[] for _ in range(num_interval)],
                   'f1': [[] for _ in range(num_interval)],
                   'recall': [[] for _ in range(num_interval)],
                   'precision': [[] for _ in range(num_interval)]}

    test_results = {'macro_test_all': [[] for _ in range(num_interval)],
                    'f1_test_all': [[] for _ in range(num_interval)],
                    'macro_test': [[] for _ in range(num_interval)],
                    'f1_test': [[] for _ in range(num_interval)]}
    if args.file_io:
        input_file = 'random_seed_10.txt'
        with open(input_file, 'r') as f:
            seeds = f.readline()
        seeds = list(map(int, seeds.split(' ')))
    else:
        seeds = [52, 574, 641, 934, 12]
        # seeds = [574]
    # seeds = [i for i in range(300, 8000)]
    # seeds = seeds * 10 # 10 runs
    seeds = seeds * 1 # 2 runs
    seed_idx_map = {i: idx for idx, i in enumerate(seeds)}
    num_run = len(seeds)
    wrapper = run_wrapper(args.dataset, args.normalization, args.cuda)

    total_AL_time = 0
    for i in range(len(seeds)):
        print('current seed is {}'.format(seeds[i]))
        val_dict, test_dict, classes_dict, cur_AL_time = wrapper.run(args.strategy, num_labeled_list=num_labeled_list,
                                                                     seed=seeds[i])

        for metric in ['micro', 'macro', 'acc', 'f1', 'recall', 'precision']:
            for j in range(len(val_dict[metric])):
                val_results[metric][j].append(val_dict[metric][j])
        
        for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
            for j in range(len(test_dict[metric])):
                test_results[metric][j].append(test_dict[metric][j])

        total_AL_time += cur_AL_time

    test_avg_results = {'macro_test_all': [0. for _ in range(num_interval)],
                    'f1_test_all': [0. for _ in range(num_interval)],
                    'macro_test': [0. for _ in range(num_interval)],
                    'f1_test': [0. for _ in range(num_interval)]}

    for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
        for j in range(len(test_results[metric])):
            test_avg_results[metric][j] = np.mean(test_results[metric][j])

    if args.model == 'GCN_update':
        dir_path = os.path.join('./10splits_10runs_results', args.dataset)
    else:
        dir_path = os.path.join('./results', args.dataset)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, '{}.txt'.format(args.strategy))
    with open(file_path, 'a') as f:
        f.write('---------datetime: %s-----------\n' % datetime.datetime.now())
        f.write(f'Budget list: {num_labeled_list}\n')
        f.write(f'learning rate: {args.lr}, epoch: {args.epochs}, weight decay: {args.weight_decay}, hidden: {args.hidden}\n')
        f.write(f'50runs using seed.txt\n')
        for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
            f.write("Test_{}_macro {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_results[metric][0])))
        

        f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
    
    if args.sample_global:
        plot(num_labeled_list, test_avg_results['macro_test_all'], args.test_percents+args.save_name+"macro_test_all_global")
        plot(num_labeled_list, test_avg_results['f1_test_all'], args.test_percents+args.save_name+"f1_test_all_global")
        plot(num_labeled_list, test_avg_results['macro_test'], args.test_percents+args.save_name+"macro_test_global")
        plot(num_labeled_list, test_avg_results['f1_test'], args.test_percents+args.save_name+"f1_test_global")
    else:
        plot(num_labeled_list, test_avg_results['macro_test_all'], args.test_percents+args.save_name+"macro_test_all")
        plot(num_labeled_list, test_avg_results['f1_test_all'], args.test_percents+args.save_name+"f1_test_all")
        plot(num_labeled_list, test_avg_results['macro_test'], args.test_percents+args.save_name+"macro_test")
        plot(num_labeled_list, test_avg_results['f1_test'], args.test_percents+args.save_name+"f1_test")
