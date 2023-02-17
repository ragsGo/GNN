import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity",  scaled=True, remove_mean=True, split=None, hot=False):
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.scaled = scaled
        self.remove_mean = remove_mean
        self.hot = hot
        self.split = split if split is not None else [(0.6, 0.8)]
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        split = ".".join(".".join(str(y) for y in x) for x in self.split)
        filename = self.raw_filename.replace("/", ":")
        return [f'data-batched-{filename}-{split}-{self.hot}-{self.num_neighbours}-{self.scaled}-{self.remove_mean}-{self.smoothing}-{self.mode}-{bits}.pt']

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
            "remove_mean": self.remove_mean,
            "hot": self.hot
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
                conv = {i: self.bits[i] for i in range(len(self.bits))}
            else:
                conv = {i: str(i+1) for i in range(len(value_columns))}
            n_neighbors = self.num_neighbours
            last_stop = 0
            data_len = df_whole.shape[0]

            for split_tuple in self.split:
                start = last_stop
                last_stop = int(split_tuple[1]*data_len) if split_tuple[1] < 1 else split_tuple[1]
                split = int(data_len*split_tuple[0]) if split_tuple[0] < 1 else split_tuple[0]

                train_set = split
                test_set = last_stop

                df_xtrain = df_whole.iloc[start:train_set,1:]
                df_ytrain = df_whole['value'][start:train_set]
                if self.hot:
                    for i in range(1, len(df_xtrain.columns)):
                        df_xtrain[i] = (df_xtrain[conv[i - 1]] == 1).astype(int)
                    for i in range(1, len(df_xtrain.columns)):
                        df_xtrain[i + len(df_xtrain.columns) - 1] = (df_xtrain[conv[i - 1]] == 2).astype(int)

                if self.remove_mean:
                    df_ytrain -= df_ytrain.mean()
                if self.scaled:
                    df_xtrain -= df_xtrain.mean()
                    df_xtrain /= df_xtrain.std()
                    df_xtrain = df_xtrain.fillna(0)

                df_xtest = df_whole.iloc[train_set:test_set, 1:]
                df_ytest = df_whole['value'][train_set:test_set]

                if self.hot:
                    for i in range(1, len(df_ytest.columns)):
                        df_ytest[i] = (df_ytest[conv[i - 1]] == 1).astype(int)
                    for i in range(1, len(df_ytest.columns)):
                        df_ytest[i + len(df_ytest.columns) - 1] = (df_ytest[conv[i - 1]] == 2).astype(int)
                if self.remove_mean:
                    df_ytest -= df_ytest.mean()
                if self.scaled:
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
                                                  n_jobs=6)
                knn_dist_graph_test = kneighbors_graph(X=df_xtest,
                                                  n_neighbors=n_neighbors,
                                                  mode=self.mode,
                                                  n_jobs=6)
                if self.smoothing == "laplacian":
                    sigma = 1
                    similarity_graph = sparse.csr_matrix(knn_dist_graph_train.shape)

                    nonzeroindices = knn_dist_graph_train.nonzero()
                    normalized = np.asarray(knn_dist_graph_train[nonzeroindices]/np.max(knn_dist_graph_train[nonzeroindices]))
                    similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
                    similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
                    graph_laplacian_s = sparse.csgraph.laplacian(csgraph=similarity_graph, normed=False)

                    graph_laplacian = graph_laplacian_s.toarray()
                    data = from_networkx(nx.from_numpy_array(graph_laplacian))
                    data.x = x_train
                    data.y = y_train
                    sigma = 1
                    similarity_graph = sparse.csr_matrix(knn_dist_graph_test.shape)
                    nonzeroindices = knn_dist_graph_test.nonzero()
                    normalized = np.asarray(knn_dist_graph_test[nonzeroindices]/np.max(knn_dist_graph_test[nonzeroindices]))
                    similarity_graph[nonzeroindices] = np.exp(-np.asarray(normalized) ** 2 / 2.0 * sigma ** 2)
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
                data.edge_index = data.edge_index.type(torch.int64)
                data.test.edge_index = data.test.edge_index.type(torch.int64)
                assert (data.edge_index.shape[0]) > 0
                assert (data.test.edge_index.shape[0]) > 0

                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_batches(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", scaled=True, remove_mean=True, split=None, hot=False, **kwargs):

    return MyOwnDataset(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, scaled=scaled, remove_mean=remove_mean, mode=mode, split=split, hot=hot)  # loader.dataset


load_data = load_data_batches
