from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import random
from scipy import sparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_networkx
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
filename = "csv-data\SNP.csv"
num_neighbours = 10


def plot_dataset(data):
    edges_raw = data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    #print(len(edges))
    labels = [n for n in list(data.y.numpy())][:len(edges)-1]

    #print(len(labels))
    #print(labels)
    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
        'node_size': 30,
        'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels, cmap=plt.cm.tab10, font_weight='bold', **options)

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

#df = df_train.loc[:, df_train.columns != 'value']
X_tsne = TSNE(learning_rate=100).fit_transform(df_train)
X_pca = PCA().fit_transform(df_train)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(df_train, y).transform(df_train)
df_scaled = (df_train-df_train.mean())/df_train.std()
n=50
#Sig = np.mat(np.eye(n)*s[:n])
#newdata = u[:,:n]
#newdata = pd.DataFrame(newdata)
u, s, v = np.linalg.svd(df_scaled, full_matrices=True)
#print(X_tsne.shape)
#print(X_r2.shape)

knn_dist_graph_train = kneighbors_graph(X=u[:,:n] ,
                                              n_neighbors=num_neighbours,
                                              mode='distance',
                                              n_jobs=6)
#knn_dist_graph_test = kneighbors_graph(X=df_test[value_columns],
                                    #n_neighbors=n_neighbors,
                                    #mode='distance',
                                    #n_jobs=6)
                                    
                                    
sigma = 1
similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)
nonzeroindices = knn_dist_graph_train.nonzero()
similarity_graph[nonzeroindices] = np.exp(
    -np.asarray(knn_dist_graph_train[nonzeroindices]) ** 2 / 2.0 * sigma ** 2)
similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
graph_laplacian = graph_laplacian_s.toarray()



#x_train = torch.tensor(X_pca.tolist(), dtype=torch.float)
ytrain = torch.tensor([n for n in y ], dtype=torch.float)
data = from_networkx(nx.from_numpy_array(graph_laplacian))
#data.x = x_train
data.y = ytrain
plot_dataset(data)                
#fig, ax = plt.subplots(figsize=(12, 10))
#sns.heatmap(graph_laplacian, ax=ax, cmap='viridis_r')
#ax.set(title='Adjacency Matrix');
#plt.show()
##print(X_tsne)
#plt.figure(figsize=(10, 5))
#plt.subplot(121)
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
#plt.subplot(122)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)



plt.show()
