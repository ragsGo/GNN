import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from random import random
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
#df = pd.read_csv(filename)
#df =  df.loc[:,1:]
#print(df.head(2))

df_scaled = (df_train-df_train.mean())/df_train.std()
#print(df_scaled.head(n=3))


u, s, v = np.linalg.svd(df_scaled, full_matrices=True)

#print(u.shape)
#print(s.shape)
#print(v.shape)

#var_explained = np.round(s**2/np.sum(s**2), decimals=3)
##var_explained
 
##sns.barplot(x=list(range(1,len(var_explained)+1)),
            ##y=var_explained, color="limegreen")
##plt.xlabel('SVs', fontsize=16)
##plt.ylabel('Percent Variance Explained', fontsize=16)
###plt.savefig('svd_scree_plot.png',dpi=100)
##plt.show()

#labels= ['SV'+str(i) for i in range(1,3)]
#svd_df = pd.DataFrame(u[:,0:2], index=df.columns[0], columns=labels)
#svd_df=svd_df.reset_index()
#svd_df.rename(columns={'index':'Target'}, inplace=True)
#print(svd_df.head())

n=2
Sig = np.mat(np.eye(n)*s[:n])
newdata = u[:,:n]
newdata = pd.DataFrame(newdata)
newdata.columns=['SVD1','SVD2']
#print(newdata.head())

newdata['target']=y_train
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('SVD 1') 
ax.set_ylabel('SVD 2') 
ax.set_title('SVD') 
targets = y_train.unique()
print(targets)
indicesToKeep = newdata['target'].isin(targets)

ax.scatter(newdata.loc[indicesToKeep, 'SVD1']
 , newdata.loc[indicesToKeep, 'SVD2']
 , c =  newdata.loc[indicesToKeep, 'target'])
 #, s = 50)
#ax.legend(targets)
#ax.grid()
plt.show()
