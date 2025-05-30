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
    parser.add_argument('--feature_path', type=str, required=True, 
                        help='Path to feature.txt')
    parser.add_argument('--train_csv', type=str, required=False, 
                        help='Path to training csv file')
    parser.add_argument('--test_csv', type=str, required=False, 
                        help='Path to testing csv file')
    
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
    # if args.test_csv is None:
    #     indices, y = load_data(args.train_csv)
        
    train_indices, y_train = load_data(args.train_csv)
    test_indices, y_test = load_data(args.test_csv)
    
    # 根据索引从特征矩阵中提取对应的特征
    X_train = features[train_indices]
    X_test = features[test_indices]
    
    # 训练 Logistic Regression 分类器
    clf = LogisticRegression(random_state=args.seed, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = clf.predict(X_test)
    

    # 保存测试索引和预测结果
    output_data = np.column_stack((test_indices + 1, y_pred))  # 索引加1恢复原始索引
    output_path = './results/lr_predictions.txt'
    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")

    # 计算评估指标
    f, recall, precision = compute_metric(y_pred, y_test)
    print("F-measure: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(f, recall, precision))

    # 保存评估指标
    metrics_path = './results/lr_metrics.txt'
    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")

    from pathlib import Path
    import os

    dataset_name = str(args.train_csv).split("/")[2]
    print(dataset_name)
    target_dir = os.path.join("../detection_results", dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, "lr_predictions.txt")
    metrics_path = os.path.join(target_dir, "lr_metrics.txt")

    np.savetxt(output_path, output_data, fmt='%d', delimiter='\t', 
               header='sample_index\tprediction', comments='')
    print(f"Saved test indices and predictions to {output_path}")

    with open(metrics_path, 'w') as f_out:
        f_out.write(f"F-measure: {f:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_path}")