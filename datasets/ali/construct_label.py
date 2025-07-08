import pandas as pd
import numpy as np


# df = pd.read_csv("./ProductLabel.txt", delimiter=' ', header=None, names=['user_no', 'label'])

# df['user_ID'] = np.arange(1, len(df) + 1)
# df = df[['user_ID', 'user_no', 'label']]
# # df['user_no'] = df['user_no']+1



# df.to_csv("./UserLabel.txt", sep=' ', index=False, header=True)

full_df = pd.DataFrame({'user_no': range(0, 36287)})

# 步骤 2：读取 label.txt 文件（假设文件中是 'id label' 格式）
label_df = pd.read_csv('Label.txt', sep=' ', header=None, names=['user_no', 'label'])

# 步骤 3：合并两个 DataFrame，未匹配的 label 设置为 -1
merged_df = pd.merge(full_df, label_df, on='user_no', how='left')
merged_df['label'] = merged_df['label'].fillna(-1).astype(int)

merged_df['user_ID'] = np.arange(1, len(merged_df) + 1)
merged_df = merged_df[['user_ID', 'user_no', 'label']]
# df['user_no'] = df['user_no']+1

merged_df.to_csv("./UserLabel.txt", sep=' ', index=False, header=True)