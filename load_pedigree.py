import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False):
        self.use_validation = False
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        return [f'pedigree-data-{filename}-{self.use_validation}-{self.smoothing}-{bits}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []
        for filename in self.raw_file_names:

            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-3)]
            labels = ["value"] + value_columns + ["father", "mum"]
            df_whole = pd.read_csv(filename)
            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]
            n_neighbors = self.num_neighbours
            df_whole['value'] = df_whole['value'] - df_whole['value'].mean()
            train_set = 2326  # 2326
            valid_set = 100 if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole[value_columns][0:train_set]
            df_ytrain = df_whole['value'][0:train_set]

            df_xtest = df_whole[value_columns][train_set:test_set]
            df_ytest = df_whole['value'][train_set:test_set]

            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)
            x_test = torch.tensor(df_xtest.values.tolist(), dtype=torch.float)
            y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)

            knn_dist_graph_train = np.zeros((train_set, train_set))
            dad_idx = df_whole["father"][:train_set].values.tolist()
            mum_idx = df_whole["mum"][:train_set].values.tolist()
            print(dad_idx)
            for i in range(train_set):
                if dad_idx[i] != 0:
                    knn_dist_graph_train[i, dad_idx[i]] = 1
                    knn_dist_graph_train[dad_idx[i], i] = 1
                if mum_idx[i] != 0:
                    knn_dist_graph_train[i, mum_idx[i]] = 1
                    knn_dist_graph_train[mum_idx[i], i] = 1

            knn_dist_graph_test = np.zeros((test_set-train_set, test_set-train_set))
            dad_idx = [x if x >= train_set else 0 for x in df_whole["father"][train_set:test_set].values.tolist()]
            mum_idx = [x if x >= train_set else 0 for x in df_whole["mum"][train_set:test_set].values.tolist()]

            for i in range(test_set-train_set):
                if dad_idx[i] != 0:
                    knn_dist_graph_test[i, dad_idx[i]+1-train_set] = 1
                    knn_dist_graph_test[dad_idx[i]+1-train_set, i] = 1
                if mum_idx[i] != 0:
                    knn_dist_graph_test[i, mum_idx[i]+1-train_set] = 1
                    knn_dist_graph_test[mum_idx[i]+1-train_set, i] = 1

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
            else:
                edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
                data = Data(x=x_train, y=y_train, edge_index=edge_index)
                
                edge_index = torch.tensor(knn_dist_graph_test.nonzero(), dtype=torch.long)
                test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
            data.test = test_data
            if self.use_validation:
                data.valid = {"x": torch.tensor(df_whole[value_columns][test_set:].values.tolist(), dtype=torch.float), "y": torch.tensor([[y] for y in df_whole['value'][test_set:].values], dtype=torch.float)}

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_pedigree(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False):

    return MyOwnDataset(".", bits=bits, raw_filename="Pedigree_Data.csv", num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation)  # loader.dataset


load_data = load_data_pedigree
