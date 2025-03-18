import torch
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import random
import pandas as pd

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输入文件路径参数
    parser.add_argument('--feature_path', type=str, required=False, 
                        help='Path to feature.txt', default="../he_amazon/data/network/product_features.txt")
    parser.add_argument('--label_txt', type=str, required=False, 
                        help='Path to training csv file', default="../he_amazon/data/network/ProductLabel.txt")
    
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

def load_data(txt_path):
    
    labels = np.zeros(3408)  # 创建一个3408维的零向量
    with open(txt_path, 'r') as f:
        for line in f:
            index, label = line.strip().split()
            labels[int(index)] = int(label)  # 在对应索引位置设置标签值
    return np.arange(3408), labels  # 返回所有索引和标签向量

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
    features = np.loadtxt(args.feature_path, delimiter='\t')
       
    indices, y = load_data(args.label_txt)
        
    # 随机选择5%的数据作为训练集
    num_samples = len(indices)
    num_train = int(num_samples * 0.05)  # 5% for training
    train_indices = np.random.choice(indices, num_train, replace=False)
    test_indices = np.array(list(set(indices) - set(train_indices)))
    
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # 根据索引从特征矩阵中提取对应的特征
    X_train = features[train_indices]
    X_test = features[test_indices]
    
    # 训练 Logistic Regression 分类器
    clf = LogisticRegression(random_state=args.seed, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = clf.predict(X_test)
    
    # 计算评估指标
    f, recall, precision = compute_metric(y_pred, y_test)
    print("F-measure: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(f, recall, precision))
