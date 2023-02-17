import numpy as np
from scipy import linalg
import pandas as pd

with open("SNP.csv") as fp:
    line = fp.readline()
    column_count = len(line.split(","))
value_columns = [str((i + 1)) for i in range(column_count - 1)]
labels = ["value"] + value_columns
train_set = 2326
df_whole = pd.read_csv("SNP.csv", names=labels)
df_xtrain = df_whole.iloc[0:train_set, 1:].values

kernel_length = 10

h = np.ones(kernel_length)

padding = np.zeros(len(value_columns) - 1, h.dtype)
first_col = np.r_[h, padding]
first_row = np.r_[h[0], padding]


H: np.ndarray = linalg.toeplitz(first_col, first_row)

for i in range(df_xtrain.shape[0]):
    val = H @ df_xtrain[0, :]
    print(val)
    print(val.shape)
    break

