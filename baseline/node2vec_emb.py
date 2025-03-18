import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import pandas as pd
import scipy.sparse as sp
from node2vec import Node2Vec
import networkx as nx

# -------------------------------
# 参数设置及数据加载函数
# -------------------------------
def parse_args():
    """
    参数解析器，同时接收特征文件、训练/测试 CSV 以及图边列表文件路径。
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
    return parser.parse_args()


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
    读取图边列表文件，每行格式为 “node1 node2”，节点编号从 1 开始
    转换为 0 索引，并假设图为无向图。
    """
    edges = []
    with open(edge_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            u, v = int(parts[0]) - 1, int(parts[1]) - 1
            edges.append((u, v))
            edges.append((v, u))  # 无向图，添加双向边
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

# 构造 NetworkX 图
def build_graph(edges, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))  # 确保所有节点都添加
    G.add_edges_from(edges)
    return G

# 运行 node2vec 生成节点嵌入
def generate_node2vec_embeddings(G, embedding_dim=128, walk_length=80, num_walks=10, p=1, q=1):
    node2vec = Node2Vec(G, dimensions=embedding_dim, walk_length=walk_length, 
                         num_walks=num_walks, p=p, q=q, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 获取所有节点的嵌入向量
    embeddings = np.zeros((G.number_of_nodes(), embedding_dim))
    for node in G.nodes():
        embeddings[node] = model.wv[str(node)]  # node2vec 的键是字符串格式

    return torch.FloatTensor(embeddings)
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
    
    # 读取图边列表并构造归一化邻接矩阵
    edges = load_edge_list(args.edge_list)
    adj = build_adj(edges, num_nodes)
    

    G = build_graph(edges, num_nodes)

    node2vec_features = generate_node2vec_embeddings(G)

    # 拼接原始特征和 node2vec 特征
    features_combined = torch.cat((features, node2vec_features), dim=1)

    print("原始特征维度:", features.shape)
    print("Node2Vec 生成的特征维度:", node2vec_features.shape)
    print("拼接后的特征维度:", features_combined.shape)

    node2vec_features_path = "../datasets/Yelp/UserFeature_node2vec.txt"
    np.savetxt(node2vec_features_path, node2vec_features.numpy(), delimiter='\t')

    # 保存拼接后的特征
    node2vec_features_combined_path = "../datasets/Yelp/UserFeature_node2vec_combined.txt"
    np.savetxt(node2vec_features_combined_path, features_combined.numpy(), delimiter='\t')