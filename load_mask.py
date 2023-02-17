import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False, split=None):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        self.bits = bits
        self.split = split if split is not None else 0.8
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        return [f'data-mask-{filename}-{self.split}-{self.use_validation}-{self.num_neighbours}-{self.smoothing}-{self.mode}-{bits}.pt']

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
            "split": self.split
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
            data_len = df_whole.shape[0]
            split = int(data_len*self.split) if self.split < 1 else self.split
            n_neighbors = self.num_neighbours
            train_set = split
            valid_set = 100 if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole.iloc[:, 1:]
            df_ytrain = df_whole['value']
            df_ytrain -= df_ytrain.mean()

            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([n for n in df_ytrain.values], dtype=torch.float)

            knn_dist_graph_train = kneighbors_graph(X=df_xtrain,
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

            else:
                edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
                data = Data(x=x_train, y=y_train, edge_index=edge_index)

            data.train_mask = range(0, train_set)
            data.test_mask = range(0, test_set)
            data.val_mask = range(0, valid_set)
            data.edge_index = data.edge_index.type(torch.int64)
            assert (data.edge_index.shape[0]) > 0
            if self.use_validation:
                data.valid = {"x": torch.tensor(df_whole.iloc[test_set:, 1:].values.tolist(), dtype=torch.float), "y": torch.tensor([[y] for y in df_whole['value'][test_set:].values], dtype=torch.float)}

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_mask(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False, split=None, ):

    return MyOwnDataset(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation, split=split)  # loader.dataset


load_data = load_data_mask
