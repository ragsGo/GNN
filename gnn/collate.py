import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

df_vals = pd.read_csv("reversed-MiceBL.csv")

d_individual = dict()

value_columns = [str((i)) for i in range(len(df_vals.columns))]
df_vals.columns = value_columns

print(df_vals['0'])
print(len(df_vals['0'].unique()))
df_vals = df_vals[['0','1','2']]

# print(df_vals.groupby(['0']).mean())

#
# print(len(df_vals['0'].unique()))
# print(df_vals[df_vals['0'].duplicated()==True])
# df_vals.groupby('0').mean().to_csv('mean.csv')


# for idx, row in df_vals.iterrows():
#     # print(d_individual)
#     if row[0] not in d_individual:
#         d_individual[row[0]] = []
#     for i in range (1 ,len(row)):
#         d_individual[row[0]].append(row[i])
#
# print("Done")
#
# x= []
# y= []
# for k, v in d_individual.items():
#     x.extend(list(itertools.repeat(k, len(v))))
#     y.extend(v)
#
#
# plt.plot(x,y,'ro')
# plt.show()
