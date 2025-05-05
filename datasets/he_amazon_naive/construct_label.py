import pandas as pd
import numpy as np


df = pd.read_csv("./ProductLabel.txt", delimiter=' ', header=None, names=['user_no', 'label'])

df['user_ID'] = np.arange(1, len(df) + 1)
df = df[['user_ID', 'user_no', 'label']]



df.to_csv("./UserLabel.txt", sep=' ', index=False, header=True)





