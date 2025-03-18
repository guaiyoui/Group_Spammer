import pandas as pd

# 读取 TXT 文件，使用空格或多个空格作为分隔符
df = pd.read_csv("UserLabel.txt", sep=r"\s+", header=0)
# 查看数据集概览
print(df.head(10))
print(df.info())

# 计算各特征的统计信息
print(df.describe())

print(df.nunique())

# 计算label列中每个值的数量
print("\nLabel value counts:")
print(df['label'].value_counts())
