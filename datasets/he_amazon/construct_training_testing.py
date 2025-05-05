import pandas as pd
import numpy as np

def process_file(file_path):
    df = pd.read_csv(file_path, delimiter=' ', names=['user_no', 'label'])
    df['user_no'] = df['user_no'].astype(int)+1
    df['label'] = df['label'].astype(int)
    # df['user_no'] = df['user_no'].astype(int)
    # df['label'] = df['label'].astype(int)
    return df


    # labels = np.zeros(3408)  # 创建一个3408维的零向量
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         index, label = line.strip(" ").split()
    #         labels[int(index)] = int(label)  # 在对应索引位置设置标签值
      
    # return pd.DataFrame({'user_no': range(3408), 'label': labels})  # 返回所有索引和标签向量




# 合并数据集
# combined_df = process_file('UserLabel.txt')
combined_df = process_file('ProductLabel.txt')

# 随机打乱数据
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算5%的数量
train_size = int(len(combined_df) * 0.50)

# 分割数据
new_train_df = combined_df.iloc[:train_size]
new_test_df = combined_df.iloc[train_size:]

# 按 user_id 排序
new_train_df = new_train_df.sort_values('user_no')
new_test_df = new_test_df.sort_values('user_no')

# Convert labels: 1 -> 0, -1 -> 1
# new_train_df['label'] = new_train_df['label'].map({1: 0, -1: 1})
# new_test_df['label'] = new_test_df['label'].map({1: 0, -1: 1})

print(new_train_df['label'].value_counts())
print(new_test_df['label'].value_counts())

# 保存到新的CSV文件，不包含索引，使用空格作为分隔符
new_train_df.to_csv('./Training_Testing/50percent/train_4.csv', index=False, sep=' ', header=False)
new_test_df.to_csv('./Training_Testing/50percent/test_4.csv', index=False, sep=' ', header=False)

print(f"处理完成。训练集大小: {len(new_train_df)}, 测试集大小: {len(new_test_df)}")