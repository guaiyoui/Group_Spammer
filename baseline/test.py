import pandas as pd
import numpy as np
import torch


feature_path = "../he_amazon/data/network/ProductLabel.txt"

# 从feature.txt中加载特征
features = np.loadtxt(feature_path, delimiter=' ')

# 获取特征维度
num_nodes, num_dimensions = features.shape

print(feature_path)
print(f"Number of nodes: {num_nodes}")
print(f"Number of dimensions: {num_dimensions}")

print(np.sum(features, axis=0))
