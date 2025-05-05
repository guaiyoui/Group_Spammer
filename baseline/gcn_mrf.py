import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import pandas as pd
import scipy.sparse as sp

# -------------------------------
# 参数解析及数据加载函数
# -------------------------------
def parse_args():
    """
    参数解析器，同时接收特征文件、训练/测试 CSV、图边列表文件路径以及 MRF 正则项权重。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--feature_path', type=str, required=True, 
                        help='Path to feature.txt')
    parser.add_argument('--train_csv', type=str, required=True, 
                        help='Path to training csv file')
    parser.add_argument('--test_csv', type=str, required=True, 
                        help='Path to testing csv file')
    parser.add_argument('--edge_list', type=str, required=True,
                        help='Path to graph edge list file')
    parser.add_argument('--lambda_mrf', type=float, default=1.0,
                        help='Weight for MRF regularization term')
    return parser.parse_args()

def compute_metric(pred, labels):
    """
    计算 F-measure、Recall 和 Precision。
    """
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == -10:
            continue
        if pred[i] == 1 and labels[i] == 1:
            tp += 1
        elif pred[i] == 0 and labels[i] == 1:
            fn += 1
        elif pred[i] == 1 and labels[i] == 0:
            fp += 1
        elif pred[i] == 0 and labels[i] == 0:
            tn += 1
        else:
            raise ValueError("the category number is incorrect")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return f, recall, precision

def load_data(csv_path):
    """
    读取 CSV 文件，每一行格式为 “sample_index label”，
    其中 sample_index 从 1 开始，因此转换为 0 索引。
    """
    df = pd.read_csv(csv_path, sep=" ", header=None, names=["sample_index", "label"])
    indices = df["sample_index"].astype(int).values - 1  # 转换为 0 索引
    labels = df["label"].astype(int).values
    return indices, labels

def load_edge_list(edge_list_path):
    """
    读取图边列表文件，每行格式为 “node1 node2”，节点编号从 1 开始，
    转换为 0 索引，并假设图为无向图（即每条边都添加双向边）。
    """
    edges = []
    with open(edge_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            u, v = int(parts[0]) - 1, int(parts[1]) - 1
            edges.append((u, v))
            edges.append((v, u))  # 添加双向边
    return edges

def build_adj(edges, num_nodes):
    """
    根据边列表构造邻接矩阵，并添加自环，再做对称归一化，
    最后转换为 PyTorch 稀疏张量。
    """
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = np.ones(len(edges))
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # 添加自环
    A = A + sp.eye(num_nodes)
    # 计算 D^(-1/2)
    rowsum = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_normalized = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
    # 转换为 torch.sparse.FloatTensor
    indices = torch.from_numpy(
        np.vstack((A_normalized.row, A_normalized.col)).astype(np.int64)
    )
    values = torch.from_numpy(A_normalized.data.astype(np.float32))
    shape = A_normalized.shape
    A_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
    return A_tensor

# -------------------------------
# GCN 模型定义
# -------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # 利用稀疏矩阵乘法： A_hat * X
        x = torch.spmm(adj, x)
        x = self.linear(x)
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid, dropout)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        x = self.gc2(x)
        return x

# -------------------------------
# MRF 正则项函数
# -------------------------------
def mrf_loss(pred_prob, edge_index):
    """
    计算 MRF 正则项：
      对于每个边 (i,j)，计算预测概率之间的 L2 距离的平方，最后求均值。
    参数：
      pred_prob: [N, num_classes] 节点的预测概率（经过 softmax）
      edge_index: [2, num_edges] 边索引张量
    """
    diff = pred_prob[edge_index[0]] - pred_prob[edge_index[1]]
    loss = torch.mean(torch.sum(diff ** 2, dim=1))
    return loss

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 从 feature.txt 中加载特征
    features_np = np.loadtxt(args.feature_path, delimiter='\t')
    num_nodes, nfeat = features_np.shape
    features = torch.FloatTensor(features_np)
    
    # 读取图边列表，构造归一化邻接矩阵，并转换为边索引 tensor（用于 MRF 损失计算）
    edges = load_edge_list(args.edge_list)
    adj = build_adj(edges, num_nodes)
    edge_index = torch.LongTensor(edges).t()  # shape: [2, num_edges]
    
    # 读取训练和测试数据
    train_indices, y_train = load_data(args.train_csv)
    test_indices, y_test = load_data(args.test_csv)
    
    # 将索引与标签转换为 torch 张量
    train_indices = torch.LongTensor(train_indices)
    test_indices = torch.LongTensor(test_indices)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # 构造 GCN 模型（这里设隐藏层维度为 16，类别数为 2）
    model = GCN(nfeat=nfeat, nhid=16, nclass=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练 GCN 模型（200 个 epoch）
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, adj)  # [num_nodes, num_classes]
        # 计算监督交叉熵损失（仅在训练节点上计算）
        loss_supervised = loss_fn(output[train_indices], y_train)
        # 计算 MRF 正则项（在整个图上计算）
        pred_prob = F.softmax(output, dim=1)
        loss_mrf = mrf_loss(pred_prob, edge_index)
        # 总损失 = 监督损失 + lambda_mrf * MRF 正则项
        loss = loss_supervised + args.lambda_mrf * loss_mrf
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print("Epoch: {:03d}, Loss: {:.4f}, CE: {:.4f}, MRF: {:.4f}".format(epoch, loss.item(), loss_supervised.item(), loss_mrf.item()))
    
    # 模型评估：在测试节点上预测并计算评价指标
    model.eval()
    output = model(features, adj)
    y_pred = output[test_indices].max(1)[1].cpu().numpy()


    # 保存测试索引和预测结果
    output_data = np.column_stack((test_indices.cpu().numpy() + 1, y_pred))  # 索引加1恢复原始索引
    output_path = './results/gcn_mrf_predictions.txt'
    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")

    f, recall, precision = compute_metric(y_pred, y_test.numpy())
    print("F-measure: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(f, recall, precision))

    # 保存评估指标
    metrics_path = './results/gcn_mrf_metrics.txt'
    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")


    from pathlib import Path
    import os

    dataset_name = str(args.train_csv).split("/")[2]
    print(dataset_name)
    target_dir = os.path.join("../detection_results", dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, "gcn_mrf_predictions.txt")
    metrics_path = os.path.join(target_dir, "gcn_mrf_metrics.txt")

    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")

    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")