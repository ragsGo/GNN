import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from statsmodels.stats.moment_helpers import cov2corr 
import numpy as np
#np.set_printoptions(threshold=np.inf)
from pprint import pprint
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
filename = "Pedigree_Data.csv"
y_train = []
y_train = np.asarray(y_train)
def plot_dataset(data):
    edges_raw = data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    #print(len(edges))
    labels = [n for n in list(data.y)][:len(edges)-1]

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

def cov_2_corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    print(A.T / d)
    A = ((A.T / d).T) / d
    #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A

def A_mat (filename):
        df = pd.read_csv(filename)

        #df = df.loc[~(df==0).all(axis=1)]
        df = df.drop(df.index[(df.father.eq(0) | df.mum.eq(0))])
        #print(df)

        df = df[:100]
        y_train = df['value'][0:100]
        s = df.father.values.flatten()
        d = df.mum.values.flatten()
        #print(s)
        if (len(s) != len(d)):
                stop("size of the father vector and mum vector are different!")
                
        
        n = len(df)
        N =  n + 1
        A =  [ [0] * N for _ in range(N)]
        #print (len(A))

        # set sires and dams
        s = (s == 0)*(N) + s
        d = (d == 0)*N + d
                    
        #print(s)
        for i in range(1,n):
                        
                    # equation for diagonals
                    
                        #print(A[5][5]/2)
                        
                        #np.fill_diagonal(A, 1 + A[s[i]][d[i]]/2)
                        A[i][i] = 1 + A[s[i]][d[i]]/2
                        
                        for j in  range(i+1,n):    # only do half of the matrix (symmetric)
                            if (j > n):
                                break
                            A[i][j] = ( A[i][s[j]] + A[i][d[j]] ) / 2  # half relationship to parents
                            A[j][i] = A[i][j]    # symmetric matrix, so copy to other off-diag
        return A
    
A = A_mat(filename)
#pprint(np.array(A))
num_neighbours = 10
knn_dist_graph_train = kneighbors_graph(X=np.array(A) ,
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



x_train = torch.tensor(np.array(A), dtype=torch.float)
#ytrain = torch.tensor([n for n in y ], dtype=torch.float)
data = from_networkx(nx.from_numpy_array(np.array(A)))
data.x = x_train
data.y = y_train
plot_dataset(data)  
plt.show()
#import numpy as np
#from scipy.sparse.csgraph import laplacian



#lap = laplacian(np.array(A))
#print(lap)
#sparse_A = csr_matrix(np.array(A))
#print(sparse_A)
##rel_mat = cov2corr(np.array(A))
##print(round(rel_mat, 4))
##print(rel_mat))
#a = cssparser_matrix((vals, (rows, cols)))
