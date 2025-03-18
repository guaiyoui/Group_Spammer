import torch
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd
import networkx as nx
import argparse

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    
    # 输入文件路径参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True, 
                        help='Path to feature.txt')
    parser.add_argument('--graph_path', type=str, required=True, 
                        help='Path to J01Network.txt')
    parser.add_argument('--train_csv', type=str, required=True, 
                        help='Path to training csv file')
    parser.add_argument('--test_csv', type=str, required=True, 
                        help='Path to testing csv file')
    parser.add_argument('--label_path', type=str, required=False, 
                        help='Path to label file')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    indices = df["sample_index"].astype(int).values - 1  # 将字符串转换为 int 后再减1
    labels = df["label"].astype(int).values
    return indices, labels


def load_graph_from_txt(graph_file, train_file, test_file, feature_file):
    """
    根据三个文本文件构建图：
    - graph_file: 每一行“x y”，表示一条边（x,y），注意文件中节点编号从1开始。
    - label_file: 每一行“index label”，表示节点的标签。
    - feature_file: 每一行“index feat1 feat2 ...”，表示节点的特征向量。
    """
    G = nx.Graph()
    
    # 加载边数据
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split()
            # 将1-index转为0-index
            x, y = int(x) - 1, int(y) - 1
            G.add_edge(x, y)
    
    # 加载标签数据
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            index = int(parts[0]) - 1  # 转换为0-index
            label = parts[1]
            # 如果标签为数字，可以转换为float或int
            try:
                label = float(label)
            except:
                pass
            if index in G.nodes:
                G.nodes[index]['label'] = label
            else:
                G.add_node(index, label=label)
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            index = int(parts[0]) - 1  # 转换为0-index
            label = parts[1]
            # 如果标签为数字，可以转换为float或int
            try:
                label = float(label)
            except:
                pass
            if index in G.nodes:
                G.nodes[index]['label'] = label
            else:
                G.add_node(index, label=label)
    
    # 加载特征数据
    with open(feature_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 整行数据即为特征向量，转换为浮点数列表
            feats = [float(x) for x in parts]
            if idx in G.nodes:
                G.nodes[idx]['feature'] = feats
            else:
                G.add_node(idx, feature=feats)
        
    return G

if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
  

    from UGFraud.Detector.SpEagle import *

    graph_file = args.graph_path
    label_file = args.label_path
    feature_file = args.feature_path
    train_file = args.train_csv
    test_file = args.test_csv

    # 根据文本文件构建图（节点编号自动转换为0-index）
    G = load_graph_from_txt(graph_file, train_file, test_file, feature_file)
    
    # 构造 ground truth（这里假设节点的 'label' 属性为评价标签）
    ground_truth = {node: data['label'] for node, data in G.nodes(data=True) if 'label' in data}
    
    # 参数设置
    numerical_eps = 1e-4
    eps = 0.2
    user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps],
                                             [numerical_eps, 1 - numerical_eps]]))
    review_product_potential = np.log(np.array([[1 - eps, eps],
                                                [eps, 1 - eps]]))
    potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
                  'r_p': review_product_potential, 'p_r': review_product_potential}
    max_iters = 6
    stop_threshold = 1e-3
    
    # 构造模型
    model = SpEagle(G, potentials, message=None, max_iters=max_iters)
    
    # 设定调度策略，采用BFS顺序
    model.schedule(schedule_type='bfs')
    
    start_iter = 0
    num_bp_iters = 4
    model.run_bp(start_iter=start_iter, max_iters=num_bp_iters, tol=stop_threshold)
    
    # 进行分类，注意这里的返回值名称与原代码保持一致
    userBelief, reviewBelief, _ = model.classify()
    
    print(userBelief, reviewBelief)


