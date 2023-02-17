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


class PlainGraph(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False, split=None, use_weights=False, algorithm="minkowski", validation_size=0.1, include_self=False, hot=False, scaled=False, remove_mean=True):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.split = split if split is not None else 0.8
        self.use_weights = use_weights
        self.algorithm = algorithm
        self.include_self = include_self
        self.validation_size = validation_size
        self.hot = hot
        self.scaled = scaled
        self.remove_mean = remove_mean
        super(PlainGraph, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if hasattr(self.data, "valid"):
            self.valid = self.data.valid

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        algorithm = self.algorithm if isinstance(self.algorithm, str) else self.algorithm.__name__
        return [f'data-{filename}-{self.split}-{self.use_validation}-{self.num_neighbours}-{self.smoothing}-{self.mode}-{self.use_weights}-{algorithm}-{self.validation_size}-{self.include_self}-{self.hot}-{self.scaled}-{self.remove_mean}-{bits}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []
        print({
            "use_validation": self.use_validation,
            "num_neighbours": self.num_neighbours,
            "smoothing": self.smoothing,
            "mode": self.mode,
            "raw_filename": self.raw_filename,
            "bits": self.bits,
            "split": self.split,
            "use_weights": self.use_weights,
            "algorithm": self.algorithm,
            "validation_size": self.validation_size,
            "include_self": self.include_self,
            "hot": self.hot,
            "scaled": self.scaled,
            "remove_mean": self.remove_mean,
        })
        for filename in self.raw_file_names:
            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i+1)) for i in range(column_count-1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)

            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]
                df_whole.columns = ["value"] + list(range(1, len(self.bits)+1))

            data_len = df_whole.shape[0]
            split = int(data_len*self.split) if self.split < 1 else self.split
            n_neighbors = self.num_neighbours
            train_set = split
            validation_size = self.validation_size if isinstance(self.validation_size, int) else int(self.validation_size*(df_whole.shape[0]-split))
            #print(validation_size)
            valid_set = validation_size if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole.iloc[0:train_set,1:]
            df_ytrain = df_whole['value'][0:train_set]
            if self.remove_mean:
                df_ytrain -= df_ytrain.mean()

            df_xtest = df_whole.iloc[train_set:test_set, 1:]
            df_ytest = df_whole['value'][train_set:test_set]
            if self.remove_mean:
                df_ytest -= df_ytest.mean()

            if self.hot:
                column_count = len(df_xtrain.columns)
                for i in range(1, column_count):
                    df_xtrain[i] = (df_xtrain[i] == 1).astype(int)
                for i in range(1, column_count):
                    df_xtrain[i + column_count - 1] = (df_xtrain[i] == 2).astype(int)
                for i in range(1, column_count):
                    df_xtest[i] = (df_xtest[i] == 1).astype(int)
                for i in range(1, column_count):
                    df_xtest[i + column_count - 1] = (df_xtest[i] == 2).astype(int)

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
                    data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()

                sigma = 1
                similarity_graph = sparse.csr_matrix(knn_dist_graph_test.shape)
                nonzeroindices = knn_dist_graph_test.nonzero()
                normalized = np.asarray(knn_dist_graph_test[nonzeroindices]/np.max(knn_dist_graph_test[nonzeroindices]))
                similarity_graph[nonzeroindices] = np.exp(np.asarray(-normalized) ** 2 / 2.0 * sigma ** 2)
                similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)
                # graph_laplacian = graph_laplacian_s.toarray() #11442,

                edge_index, edge_weight = from_scipy_sparse_matrix(graph_laplacian_s)
                # print(edge_weight.mean())
                # print(edge_weight.shape)from scipy import sparse
                #
                # sn.heatmap(df_whole.iloc[: , 1:])
                # plt.show()
                # printsknn_dist_graph_train
                # data = from_networkx(nx.from_numpy_array(graph_laplacian))
                test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
                # test_data = from_networkx(nx.from_numpy_array(graph_laplacian))
                # test_data.x = x_test
                # test_data.y = y_test
                if self.use_weights:
                    # test_graph = knn_dist_graph_test/np.max(knn_dist_graph_test)
                    # test_graph[nonzeroindices] = 1 - test_graph[nonzeroindices]
                    # test_data.edge_weight = torch.tensor(test_graph.todense(), dtype=torch.float)
                    test_data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()
            else:
                edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_train)
                # edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
                data = Data(x=x_train, y=y_train, edge_index=edge_index)
                if self.use_weights:
                    # nonzeroindices = knn_dist_graph_train.nonzero()
                    # graph = knn_dist_graph_train/np.max(knn_dist_graph_train)
                    # graph[nonzeroindices] = 1 - graph[nonzeroindices]
                    # data.edge_weight = torch.tensor(graph.todense(), dtype=torch.float)
                    data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()

                edge_index, edge_weight = from_scipy_sparse_matrix(knn_dist_graph_test)
                # edge_index = torch.tensor(knn_dist_graph_test.nonzero(), dtype=torch.long)
                test_data = Data(x=x_test, y=y_test, edge_index=edge_index)
                if self.use_weights:
                    # nonzeroindices = knn_dist_graph_test.nonzero()
                    # test_graph = knn_dist_graph_test/np.max(knn_dist_graph_test)
                    # test_graph[nonzeroindices] = 1 - test_graph[nonzeroindices]
                    # test_data.edge_weight = torch.tensor(test_graph.todense(), dtype=torch.float)
                    test_data.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()

            data.test = test_data
            data.edge_index = data.edge_index.type(torch.int64)
            data.test.edge_index = data.test.edge_index.type(torch.int64)
            assert (data.edge_index.shape[0]) > 0
            assert (data.test.edge_index.shape[0]) > 0
            if self.use_validation:
                df_xvalid = df_whole.iloc[test_set:, 1:]
                df_yvalid = df_whole['value'][test_set:]
                df_yvalid -= df_yvalid.mean()
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
                        valid.edge_weight = 1-abs(edge_weight)/max(edge_weight).float()

                data.valid = valid

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False, split=None, use_weights=False, algorithm="minkowski", validation_size=0.1, hot=False, scaled=False, remove_mean=True, **_):

    return PlainGraph(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation, split=split, use_weights=use_weights, algorithm=algorithm, validation_size=validation_size, hot=hot, scaled=scaled, remove_mean=remove_mean)  # loader.dataset
