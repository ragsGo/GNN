from random import random

import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

from load_rels import create_parent_edges
from load_siblings import create_sibling_edges


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False, replace_twos=True):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.replace_twos = replace_twos
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        return [f'data-two-{filename}-{self.use_validation}-{self.num_neighbours}-{self.smoothing}-{self.mode}-{self.replace_twos}-{bits}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []
        for filename in self.raw_file_names:

            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)
            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]
            df_whole['value'] = df_whole['value'] - df_whole['value'].mean()

            if self.replace_twos:
                df_whole.replace(2, 0)
            n_neighbors = self.num_neighbours
            train_set = 2326
            valid_set = 100 if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole.iloc[0:train_set,1:]
            df_ytrain = df_whole['value'][0:train_set]

            df_xtest = df_whole.iloc[train_set:test_set, 1:]
            df_ytest = df_whole['value'][train_set:test_set]

            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)
            x_test = torch.tensor(df_xtest.values.tolist(), dtype=torch.float)
            y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)
            knn_dist_graph_train = kneighbors_graph(X=df_xtrain,
                                              n_neighbors=n_neighbors,
                                              mode=self.mode,
                                              n_jobs=6)
            knn_dist_graph_test = kneighbors_graph(X=df_xtest,
                                              n_neighbors=n_neighbors,
                                              mode=self.mode,
                                              n_jobs=6)
            
            if self.smoothing == "laplacian":
                sigma = 1
                similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)
                nonzeroindices = knn_dist_graph_train.nonzero()
                similarity_graph[nonzeroindices] = np.exp(
                    -np.asarray(knn_dist_graph_train[nonzeroindices]) ** 2 / 2.0 * sigma ** 2)
                similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
                graph_laplacian = graph_laplacian_s.toarray()

                data = from_networkx(nx.from_numpy_array(graph_laplacian))
                data.x = x_train
                data.y = y_train
                data.edge_index = (data.edge_index, torch.tensor(create_parent_edges("Pedigree_Data.csv", slice(0, train_set)), dtype=torch.float))

                sigma = 1
                similarity_graph = sparse.csr_matrix(knn_dist_graph_test.shape)
                nonzeroindices = knn_dist_graph_test.nonzero()
                similarity_graph[nonzeroindices] = np.exp(
                    -np.asarray(knn_dist_graph_test[nonzeroindices]) ** 2 / 2.0 * sigma ** 2)
                similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
                graph_laplacian = graph_laplacian_s.toarray()

                test_data = from_networkx(nx.from_numpy_array(graph_laplacian))
                test_data.x = x_test
                test_data.y = y_test
                test_data.edge_index = (test_data.edge_index, torch.tensor(create_parent_edges("Pedigree_Data.csv", slice(train_set, test_set)), dtype=torch.float))
            else:
                edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
                data = Data(x=x_train, y=y_train, edge_index=edge_index)
                data.edge_index = (data.edge_index, torch.tensor(create_parent_edges("Pedigree_Data.csv", slice(0, train_set)), dtype=torch.float))

                edge_index = torch.tensor(knn_dist_graph_test.nonzero(), dtype=torch.long)
                test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
                test_data.edge_index = (test_data.edge_index, torch.tensor(create_parent_edges("Pedigree_Data.csv", slice(train_set, test_set)), dtype=torch.float))


            data.test = test_data
            if self.use_validation:
                data.valid = {"x": torch.tensor(df_whole.iloc[test_set:, 1:].values.tolist(), dtype=torch.float), "y": torch.tensor([[y] for y in df_whole['value'][test_set:].values], dtype=torch.float)}

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_two(filename, bits=None, num_neighbours=10, smoothing="none", mode="distance", use_validation=False):

    return MyOwnDataset(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation)  # loader.dataset


load_data = load_data_two
