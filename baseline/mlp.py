import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of MLP layers')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Size of hidden layers in MLP')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输入文件路径参数
    parser.add_argument('--feature_path', type=str, required=True, 
                        help='Path to feature.txt')
    parser.add_argument('--train_csv', type=str, required=True, 
                        help='Path to training csv file')
    parser.add_argument('--test_csv', type=str, required=True, 
                        help='Path to testing csv file')
    
    return parser.parse_args()

def compute_metric(pred, labels):
    """
    计算 F-measure、Recall 和 Precision。
    """
    f = f1_score(labels, pred)
    recall = recall_score(labels, pred)
    precision = precision_score(labels, pred)
    return f, recall, precision

def load_data(csv_path):
    """
    读取 CSV 文件，每一行格式为 “sample_index label”，
    其中 sample_index 从 1 开始，因此转换为 0 索引。
    """
    df = pd.read_csv(csv_path, sep=" ", header=None, names=["sample_index", "label"])
    print(df)
    indices = df["sample_index"].astype(int).values - 1  # 将字符串转换为 int 后再减1
    labels = df["label"].astype(int).values
    return indices, labels

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers, dropout):
        super(MLPClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, 2))  # 假设二分类任务
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
    
    # 读取训练和测试数据
    train_indices, y_train = load_data(args.train_csv)
    test_indices, y_test = load_data(args.test_csv)
    
    # 根据索引从特征矩阵中提取对应的特征
    X_train = features[train_indices]
    X_test = features[test_indices]
    
    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 创建数据加载器
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # 定义 MLP 分类器
    input_dim = X_train.shape[1]
    clf = MLPClassifier(input_dim, args.hidden_size, args.n_layers, args.dropout)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
    
    # 训练模型
    clf.train()
    for epoch in range(1000):  # 设置训练周期
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
    
    # 在测试集上进行预测
    clf.eval()
    with torch.no_grad():
        outputs = clf(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
    
    # 保存测试索引和预测结果
    output_data = np.column_stack((test_indices + 1, y_pred))  # 索引加1恢复原始索引
    output_path = './results/mlp_predictions.txt'
    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")
    
    # 计算评估指标
    f, recall, precision = compute_metric(y_pred.numpy(), y_test)
    print("F-measure: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(f, recall, precision))

    # 保存评估指标
    metrics_path = './results/mlp_metrics.txt'
    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")

    from pathlib import Path
    import os

    dataset_name = str(args.train_csv).split("/")[2]
    print(dataset_name)
    target_dir = os.path.join("../detection_results", dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, "mlp_predictions.txt")
    metrics_path = os.path.join(target_dir, "mlp_metrics.txt")

    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")

    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")