import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from scipy import sparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_networkx



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


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, raw_filename, num_neighbours=10, smoothing="laplacian", mode="distance", use_validation=False):
        self.use_validation = use_validation
        self.mode = mode
        self.smoothing = smoothing
        self.num_neighbours = num_neighbours
        self.raw_filename = raw_filename
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        filename = self.raw_filename.replace("/", ":")
        return [f'merge-data-{filename}-{self.num_neighbours}-{self.smoothing}-{self.mode}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []

        filename = "Pedigree_Data.csv"
        df = pd.read_csv(filename)

        df = df.loc[~(df == 0).all(axis=1)]
        print(df.columns)
        df['value'] = df['value'] - df['value'].mean()

        df = df[:2326]

        rel_father = list(zip(df.value, df.father))
        rel_mum = list(zip(df.value, df.mum))

        G_father = nx.Graph()
        G_father.add_edges_from(rel_father)

        rel_parents = list(zip(df.value, df.father, df.mum))

        G_mum = nx.Graph()
        G_mum.add_edges_from(rel_mum)

        H = nx.compose(G_father, G_mum)

        rel_mat = nx.to_numpy_array(H)

        knn_dist_graph_train = kneighbors_graph(X=rel_mat,
                                                n_neighbors=self.num_neighbours,
                                                mode=self.mode,
                                                n_jobs=6)

        test_df = df[2326:]

        t_rel_father = list(zip(test_df.value, test_df.father))
        t_rel_mum = list(zip(test_df.value, test_df.mum))

        t_rel_parents = list(zip(test_df.value, test_df.father, test_df.mum))

        t_G_father = nx.Graph()
        t_G_father.add_edges_from(t_rel_father)

        t_G_mum = nx.Graph()
        t_G_mum.add_edges_from(t_rel_mum)

        t_H = nx.compose(t_G_father, t_G_mum)

        t_rel_mat = nx.to_numpy_array(t_H)
        print(len(rel_mat))

        # knn_dist_graph_test = kneighbors_graph(X=t_rel_mat,
        #                                         n_neighbors=num_neighbours,
        #                                         mode=mode,
        #                                         n_jobs=6)

        sigma = 1
        similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)
        nonzeroindices = knn_dist_graph_train.nonzero()
        similarity_graph[nonzeroindices] = np.exp(
            -np.asarray(knn_dist_graph_train[nonzeroindices]) ** 2 / 2.0 * sigma ** 2)
        similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
        graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
        graph_laplacian = graph_laplacian_s.toarray()

        y_train = torch.tensor([[n] for n in df.value.values], dtype=torch.float)
        x_train = torch.tensor(rel_mat, dtype=torch.float)
        data = from_networkx(nx.from_numpy_array(rel_mat))
        data.x = x_train
        data.y = y_train

        y_test = torch.tensor([[n] for n in test_df.value.values], dtype=torch.float)
        x_test = torch.tensor(t_rel_mat, dtype=torch.float)
        data.test = from_networkx(nx.from_numpy_array(t_rel_mat))
        data.test.x = x_test
        data.test.y = y_test

        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_merge(filename, num_neighbours=10, smoothing="laplacian", mode="distance", use_validation=False):

    return MyOwnDataset(".", raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation)


load_data = load_data_merge
