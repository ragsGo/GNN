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
def plot_dataset(data):
    edges_raw = data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    #print(len(edges))
    #labels = [n for n in list(data.y)][:len(edges)-1]

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
    nx.draw(G, with_labels=False, cmap=plt.cm.tab10, font_weight='bold', **options)

# key structure to store a binary tree node
class Node:
	def __init__(self, key, left = None, right = None):
		self.key = key
		self.left = left
		self.right = right

# Utility function to print binary tree nodes in-order fashion
def inorder(node):
	if node:
		inorder(node.left)
		print(node.key, end = ' ')
		inorder(node.right)

# Function to construct a binary tree
# from specified ancestor matrix
def constructBT(mat):

	# get number of rows in the matrix
	N = len(mat)

	# create an empty multi-dict
	dict = {}

	# Use sum as key and row numbers as values in the multi-dict
	for i in range(N):

		# find the sum of the current row
		total = sum(mat[i])

		# insert the sum and row number into the dict
		dict.setdefault(total, []).append(i)

	# node[i] will store node for i in constructed tree
	node = [Node(-1)] * N
	last = 0

	# the value of parent[i] is true if parent is set for i'th node
	parent = [False] * N

	# Traverse the dictionary in sorted order (default behavior)
	for key in dict.keys():
		for row in dict.get(key):
			last = row
			
			# create a new node
			node[row] = Node(row)

			# if leaf node, do nothing
			if key == 0:
				continue

			# traverse row
			for i in range(N):
			
				# do if parent is not set and ancestor exits
				if not parent[i] and mat[row][i] == 1:
				
					# check for the unoccupied node
					if node[row].left is None:
						node[row].left = node[i]
					else:
						node[row].right = node[i]

					# set parent for i'th node
					parent[i] = True

	# last processed node is the root
	return node[last]


###df_qtl = pd.read_csv('SNP.csv', header=None)
##with open('SNP.csv') as fp:
                ##line = fp.readline()
                ##column_count = len(line.split(","))
##value_columns = [str((i+1)) for i in range(column_count-1)]
##labels = ["value"] + value_columns
##df_qtl = pd.read_csv('SNP.csv', names=labels)

##filename = "QTLMAS2010pedigree.txt"
##df_whole = pd.read_csv(filename,sep=" ", header=None)

##df = df_whole.iloc[0: , :-2]

##df.columns = ['value', 'father', 'mum']

###print(df_qtl)
###print(df)
#df = df.loc[~(df==0).all(axis=1)]

##df_data = pd.concat([df_qtl, df[['father', 'mum']]], axis=1)

###print(df_data.columns)
##df_data.to_csv('Pedigree_Data.csv', index=False)



filename = "Pedigree_Data.csv"
df = pd.read_csv(filename)

#df = df_whole.iloc[: , :-2]

##df.columns = ['value', 'father', 'mum']
##df =df.drop(df.loc[df[['mum', 'father']]==0].index)
df = df.loc[~(df==0).all(axis=1)]
print(df.columns)

df = df[:2326]

rel_father = list(zip(df.value, df.father))
rel_mum = list(zip(df.value, df.mum))
#print (np.unique(df.father.values))
#print (np.unique(df.mum.values))

#prints
rel_parents = list(zip(df.value, df.father,  df.mum))
y_train = df['value']
#print(rel_father)
#print(rel_parents)
G_father = nx.Graph()
G_father.add_edges_from(rel_father)

#nx.draw(G_father, 
        #node_color='lightblue', 
        #with_labels=True,
        #)

#plt.show()

G_mum = nx.Graph()
G_mum.add_edges_from(rel_mum)


#nx.draw(G_mum, 
        #node_color='pink', 
        #with_labels=True,
        #)

#plt.show()
#start = time.time()
H = nx.compose(G_father,G_mum)

#print (nx.to_numpy_matrix(H))
rel_mat = nx.to_numpy_array(H)
#nx.draw(H, 
        #node_color='lightgreen', 
        #with_labels=True,
        #)
#stop = time.time()
#print(stop-start)
plt.show()



num_neighbours = 10
knn_dist_graph_train = kneighbors_graph(X=rel_mat ,
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


x_train = torch.tensor(rel_mat, dtype=torch.float)
#ytrain = torch.tensor([n for n in y ], dtype=torch.float)
data = from_networkx(nx.from_numpy_array(rel_mat))
data.x = x_train
data.y = y_train
plot_dataset(data)  
plt.show()
