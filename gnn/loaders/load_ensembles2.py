import random

import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
import seaborn as sn
import matplotlib.pyplot as plt


class EnsembleGraph(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity",  scaled=True, remove_mean=True, split=5, hot=False, algorithm="euclidean", add_full=False, full_split=None, train_size=0.7, use_weights=False, use_validation=False, validation_size=0.1, include_self=False, separate_sets=True):
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.scaled = scaled
        self.remove_mean = remove_mean
        self.hot = hot
        self.algorithm = algorithm
        self.split = split
        self.train_size = train_size
        self.add_full = add_full
        self.full_split = full_split
        self.use_weights = use_weights
        self.use_validation = use_validation
        self.validation_size = validation_size
        self.include_self = include_self
        self.separate_sets = separate_sets
        super(EnsembleGraph, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # if self.use_validation:
        #     self.valid = self.data[-1]

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        return [f'data-ensemble2-{filename}-{self.split}-{self.add_full}-{self.full_split}-{self.train_size}-{self.hot}-{self.algorithm}-{self.num_neighbours}-{self.scaled}-{self.remove_mean}-{self.smoothing}-{self.mode}-{self.use_weights}-{self.use_validation}-{self.include_self}-{self.separate_sets}-{bits}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []
        print({
            "num_neighbours": self.num_neighbours,
            "smoothing": self.smoothing,
            "mode": self.mode,
            "raw_filename": self.raw_filename,
            "bits": self.bits,
            "split": self.split,
            "scaled": self.scaled,
            "add_full": self.add_full,
            "remove_mean": self.remove_mean,
            "train_size": self.train_size,
            "use_weights": self.use_weights,
            "hot": self.hot,
            "algorithm": self.algorithm,
            "use_validation": self.use_validation,
            "validation_size": self.validation_size,
            "include_self": self.include_self
        })
        for filename in self.raw_file_names:
            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)
            if self.use_validation:
                valid_size = self.validation_size if isinstance(self.validation_size, int) else int(df_whole.shape[0]*self.validation_size)
                df_whole = df_whole[valid_size:]
                df_valid = df_whole[:valid_size]


            full_split = int(self.full_split*len(df_whole)) if isinstance(self.full_split, float) else self.full_split

            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]
                conv = {i: self.bits[i] for i in range(len(self.bits))}
            else:
                conv = {i: str(i+1) for i in range(len(value_columns))}
            n_neighbors = self.num_neighbours

            train_size = self.train_size if self.train_size < 1 else self.train_size/df_whole.shape[0]


            if self.separate_sets:
                whole_train = df_whole.sample(frac=train_size).reset_index(drop=True)
                whole_test = df_whole.loc[~df_whole.index.isin(whole_train.index)].reset_index(drop=True)

            else:
                # shouldn't be used, but prevents complaints from linter
                whole_train = df_whole
                whole_test = df_whole


            for split in range(self.split + (1 if self.add_full else 0)):
                if split == self.split and self.add_full:
                    if full_split is None:
                        df_train = df_whole.sample(frac=train_size, random_state=split).reset_index(drop=True)
                        df_test = df_whole.copy().reset_index(drop=True)
                    else:
                        df_train = df_whole[:full_split].reset_index(drop=True)
                        df_test = df_whole[full_split:].reset_index(drop=True)

                else:
                    if self.separate_sets:
                        df_train = whole_train.sample(frac=0.5, random_state=split).reset_index(drop=True)
                        df_test = whole_test
                    else:
                        if full_split is None:
                            df_train = df_whole.sample(frac=train_size, random_state=split).reset_index(drop=True)
                            df_test = df_whole.loc[~df_whole.index.isin(df_train.index)].reset_index(drop=True)
                        else:
                            df_whole_less_full = df_whole[:full_split]
                            train_size = self.train_size if self.train_size < 1 else self.train_size/df_whole_less_full.shape[0]
                            df_train = df_whole_less_full.sample(frac=train_size, random_state=split).reset_index(drop=True)
                            df_test = df_whole_less_full.loc[~df_whole_less_full.index.isin(df_train.index)].reset_index(drop=True)
                df_xtrain = df_train.iloc[:, 1:]
                df_ytrain = df_train['value']

                df_xtest = df_test.iloc[:, 1:]
                df_ytest = df_test['value']

                if self.hot:
                    for i in range(1, len(df_xtrain.columns)):
                        df_xtrain[i] = (df_xtrain[conv[i - 1]] == 1).astype(int)
                    for i in range(1, len(df_xtrain.columns)):
                        df_xtrain[i + len(df_xtrain.columns) - 1] = (df_xtrain[conv[i - 1]] == 2).astype(int)
                    for i in range(1, len(df_xtest.columns)):
                        df_xtest[i] = (df_xtest[conv[i - 1]] == 1).astype(int)
                    for i in range(1, len(df_xtest.columns)):
                        df_xtest[i + len(df_xtest.columns) - 1] = (df_xtest[conv[i - 1]] == 2).astype(int)

                if self.remove_mean:
                    df_ytrain -= df_ytrain.mean()
                    df_ytest -= df_ytest.mean()
                if self.scaled:
                    df_xtrain -= df_xtrain.mean()
                    df_xtrain /= df_xtrain.std()
                    df_xtrain = df_xtrain.fillna(0)

                    df_xtest -= df_xtest.mean()
                    df_xtest /= df_xtest.std()
                    df_xtest = df_xtest.fillna(0)

                x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
                y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)

                x_test = torch.tensor(df_xtest.values.tolist(), dtype=torch.float)
                y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)

                knn_dist_graph_train = kneighbors_graph(X=df_xtrain,
                                                  n_neighbors=n_neighbors,
                                                  mode=self.mode,
                                                  metric=self.algorithm,
                                                  include_self=self.include_self,
                                                  n_jobs=6)

                knn_dist_graph_test = kneighbors_graph(X=df_xtest,
                                                  n_neighbors=n_neighbors,
                                                  mode=self.mode,
                                                  metric=self.algorithm,
                                                  include_self=self.include_self,
                                                  n_jobs=6)

                if self.smoothing == "laplacian":
                    sigma = 1
                    similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)

                    nonzeroindices = knn_dist_graph_train.nonzero()
                    normalized = np.asarray(knn_dist_graph_train[nonzeroindices]/np.max(knn_dist_graph_train[nonzeroindices]))
                    similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
                    similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)

                    # graph_laplacian = graph_laplacian_s.toarray()
                    # print("graph_laplacian index", from_scipy_sparse_matrix(graph_laplacian_s))
                    edge_index, edge_weight = from_scipy_sparse_matrix(graph_laplacian_s)

                    # data = from_netwsorkx(nx.from_numpy_array(graph_laplacian))
                    data = Data(x=x_train, y=y_train, edge_index=edge_index)
                    # data.x = x_train
                    # data.y = y_train
                    if self.use_weights:
                        # graph = knn_dist_graph_train/np.max(knn_dist_graph_train)
                        # graph[nonzeroindices] = 1 - graph[nonzeroindices]
                        # data.edge_weight = torch.tensor(graph.todense(), dtype=torch.float)
                        data.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()

                    similarity_graph = sparse.csr_matrix(knn_dist_graph_test.shape)

                    nonzeroindices = knn_dist_graph_test.nonzero()
                    normalized = np.asarray(knn_dist_graph_test[nonzeroindices]/np.max(knn_dist_graph_test[nonzeroindices]))
                    similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
                    similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)

                    edge_index, edge_weight = from_scipy_sparse_matrix(graph_laplacian_s)
                    # print(type(edge_weight))
                    # data = from_networkx(nx.from_numpy_array(graph_laplacian))
                    test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
                    # test_data = from_networkx(nx.from_numpy_array(graph_laplacian))
                    # test_data.x = x_test
                    # test_data.y = y_test
                    if self.use_weights:
                        # test_graph = knn_dist_graph_test/np.max(knn_dist_graph_test)
                        # test_graph[nonzeroindices] = 1 - test_graph[nonzeroindices]
                        # test_data.edge_weight = torch.tensor(test_graph.todense(), dtype=torch.float)
                        test_data.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()
                else:
                    edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_train)
                    # edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
                    data = Data(x=x_train, y=y_train, edge_index=edge_index)
                    if self.use_weights:
                        # nonzeroindices = knn_dist_graph_train.nonzero()
                        # graph = knn_dist_graph_train/np.max(knn_dist_graph_train)
                        # graph[nonzeroindices] = 1 - graph[nonzeroindices]
                        # data.edge_weight = torch.tensor(graph.todense(), dtype=torch.float)
                        data.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()

                    edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_test)
                    # edge_index = torch.tensor(knn_dist_graph_test.nonzero(), dtype=torch.long)
                    test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
                    if self.use_weights:
                        # nonzeroindices = knn_dist_graph_test.nonzero()
                        # test_graph = knn_dist_graph_test/np.max(knn_dist_graph_test)
                        # test_graph[nonzeroindices] = 1 - test_graph[nonzeroindices]
                        # test_data.edge_weight = torch.tensor(test_graph.todense(), dtype=torch.float)
                        test_data.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()

                data.test = test_data
                data.edge_index = data.edge_index.type(torch.int64)
                data.test.edge_index = data.test.edge_index.type(torch.int64)
                assert (data.edge_index.shape[0]) > 0
                assert (data.test.edge_index.shape[0]) > 0
                data_list.append(data)
            if self.use_validation:
                df_xvalid = df_valid.reset_index(drop=True)
                df_xvalid = df_valid.iloc[:, 1:]
                df_yvalid = df_valid['value']

                if self.remove_mean:
                    df_yvalid -= df_yvalid.mean()
                if self.scaled:
                    df_xvalid -= df_xvalid.mean()
                    df_xvalid /= df_xvalid.std()
                    df_xvalid = df_xvalid.fillna(0)
                x_valid = torch.tensor(df_xvalid.values.tolist(), dtype=torch.float)
                y_valid = torch.tensor([[n] for n in df_yvalid.values], dtype=torch.float)

                knn_dist_graph_valid = kneighbors_graph(X=df_xvalid,
                                                        n_neighbors=n_neighbors,
                                                        mode=self.mode,
                                                        metric=self.algorithm,
                                                        include_self=self.include_self,
                                                        n_jobs=6)
                if self.smoothing == "laplacian":
                    sigma = 1
                    similarity_graph = sparse.csr_matrix(knn_dist_graph_valid.shape)
                    nonzeroindices = knn_dist_graph_valid.nonzero()
                    normalized = np.asarray(
                        knn_dist_graph_valid[nonzeroindices] / np.max(knn_dist_graph_valid[nonzeroindices]))

                    similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
                    similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)

                    edge_index, edge_weight = from_scipy_sparse_matrix(graph_laplacian_s)
                    valid = Data(x=x_valid, y=y_valid, edge_index=edge_index)
                    if self.use_weights:
                        valid.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()
                else:
                    edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_valid)
                    valid = Data(x=x_valid, y=y_valid, edge_index=edge_index)
                    if self.use_weights:
                        valid.edge_weight = 1 - abs(edge_weight) / max(edge_weight).float()
                valid.test = valid
                data_list.append(valid)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_ensembles2(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", scaled=True, remove_mean=True, split=None, hot=False, algorithm="euclidean", add_full=False, train_size=0.7, use_weights=False, use_validation=False, validation_size=0.1, full_split=None, include_self=False, separate_sets=True, **kwargs):

    return EnsembleGraph(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, scaled=scaled, remove_mean=remove_mean, mode=mode, split=split, hot=hot, algorithm=algorithm, add_full=add_full, train_size=train_size, use_weights=use_weights, use_validation=use_validation, validation_size=validation_size, full_split=full_split, include_self=include_self, separate_sets=separate_sets)  # loader.dataset


load_data = load_data_ensembles2