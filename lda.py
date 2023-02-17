from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

from random import random

def get_cmap(n, name='hsv'):
    return (plt.cm.get_cmap(name, n))
filename = "SNP.csv"


with open(filename) as fp:
            line = fp.readline()
            column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)



y_train = df_whole['value'][0:2326]
#print(res_df.shape[1])
#Xtest= df_whole.iloc[2327:df_whole.shape[0],1:df_whole.shape[1]]
#ytest = df_whole['value'][2326:df_whole.shape[0]]
df_train = df_whole.iloc[0:2326,1:df_whole.shape[1]]

#df_train = df_whole.loc[mask]

#df_test = df_whole.loc[[not x for x in mask]]
y = y_train.values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1
targets = np.unique(y)
df_train['labels'] = y
indicesToKeep = df_train['labels'].isin(targets)
pca = PCA(n_components=2)
X_r = pca.fit(df_train).transform(df_train)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(df_train, y).transform(df_train)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
 % str(pca.explained_variance_ratio_))
plt.figure()
data = df_train.loc[indicesToKeep, 'labels'].values
colors  = get_cmap(len(data))
#print(colors)
lw = 2
for i, target_name in zip(targets, targets):
 plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=colors(i), alpha=.8, lw=lw,
 label=target_name)
#plt.legend(loc='lower left', ncol=20, shadow=False, scatterpoints=1)
plt.title('PCA of  dataset')
plt.figure()
for i, target_name in zip(targets, targets):
 plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=colors(i),
 label=target_name)
#plt.legend(loc='lower left', ncol=20, shadow=False, scatterpoints=1)
plt.title('LDA of  dataset')
plt.show()
