import pandas as pd
import numpy as np


df = pd.read_csv("./ProductNetwork.txt", delimiter=' ', header=None, names=['user_index_1', 'user_index_2'])
df['user_index_1'] += 1
df['user_index_2'] += 1

df.to_csv("./ProductNetwork_processed.txt", sep=' ', index=False, header=False)





