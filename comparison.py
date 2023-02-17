import numpy as np
import pandas as pd
import time

# For plotting
import plotly.io as plt_io
import plotly.graph_objects as go
#%matplotlib inline

#PCA
from sklearn.decomposition import PCA
#TSNE
from sklearn.manifold import TSNE
#UMAP
import umap
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder

## Standardizing the data
from sklearn.preprocessing import StandardScaler
filename = "SNP.csv"
def plot_2d(component1, component2):
    
    fig = go.Figure(data=go.Scatter(
        x = component1,
        y = component2,
        mode='markers',
        marker=dict(
            size=20,
            color=y, #set color equal to a variable
            colorscale='Rainbow', # one of plotly colorscales
            showscale=True,
            line_width=1
        )
    ))
    fig.update_layout(margin=dict( l=100,r=100,b=100,t=100),width=2000,height=1200)                 
    fig.layout.template = 'plotly_dark'
    fig.show()

def plot_3d(component1,component2,component3):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10,
            color=y,                # set color to an array/list of desired values
            colorscale='Rainbow',   # choose a colorscale
            opacity=1,
            line_width=1
        )
    )])
    #print('fig ==', fig)
# tight layout
    fig.update_layout(margin=dict(l=50,r=50,b=50,t=50),width=1800,height=1000)
    fig.layout.template = 'plotly_dark'
    #print('here')
    fig.show()
    

with open(filename) as fp:
            line = fp.readline()
            column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)
#mask = [True if random() > 0.8 else False for _ in range(len(df_whole))]
#ytot = df_whole.iloc[:,0]-np.mean(res_df.iloc[:,0])

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

#df_train['yclass'] = y
#df_train = df_train[(df_train['yclass'] >= 600) & (df_train['yclass'] <= 1000)]
#y = df_train['yclass'].values
#df = df_train.loc[:, df_train.columns != 'yclass']
#print(df.columns)
#unique, counts = np.unique(y, return_counts=True)
#print(dict(zip(unique, counts)))

## Standardizing the data
df = StandardScaler().fit_transform(df_train)

#pca = PCA(n_components=3)
#principalComponents = pca.fit_transform(df)

#principal = pd.DataFrame(data = principalComponents
             #, columns = ['principal component 1', 'principal component 2','principal component 3'])
#print('principal ==;', principal)
##

#tsne = TSNE(learning_rate=100).fit_transform(df)

reducer = umap.UMAP(random_state=42,n_components=3)
embedding = reducer.fit_transform(df)

#X_LDA = LDA(n_components=3).fit_transform(df_train,y)
#u, s, v = np.linalg.svd(df, full_matrices=True)
n=2
#newdata = u[:,:n]
plot_2d(reducer.embedding_[:, 0],reducer.embedding_[:, 1])

#plot_3d(reducer.embedding_[:, 0],reducer.embedding_[:, 1],reducer.embedding_[:, 2])
