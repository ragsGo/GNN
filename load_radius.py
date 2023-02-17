import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import radius_neighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, raw_filename, bits=None, num_radius=75, smoothing="laplacian", mode="connectivity", metric="minkowski", use_validation=False):
        self.use_validation = use_validation
        self.bits = bits
        self.metric = metric
        self.mode = mode
        self.smoothing = smoothing
        self.num_radius = num_radius
        self.raw_filename = raw_filename
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        bits = "whole" if self.bits is None else "bits"
        filename = self.raw_filename.replace("/", ":")
        return [f'radius-data-{filename}-{self.mode}-{self.metric}-{self.num_radius}-{self.smoothing}-{bits}.pt']

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
            n_radius = self.num_radius
            df_whole['value'] = df_whole['value'] - df_whole['value'].mean()

            df_xtrain = df_whole.iloc[0:2326, 1:]
            df_ytrain = df_whole['value'][0:2326]

            df_xtest= df_whole.iloc[2327:,1:]
            df_ytest = df_whole['value'][2327:]
            
            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)
            x_test = torch.tensor(df_xtest.values.tolist(), dtype=torch.float)
            y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)
            knn_dist_graph_train = radius_neighbors_graph(X=df_xtrain,
                                              radius=n_radius,
                                              metric=self.metric,
                                              mode=self.mode,
                                              n_jobs=6)
            knn_dist_graph_test = radius_neighbors_graph(X=df_xtest,
                                              radius=n_radius,
                                              metric=self.metric,
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
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_radius(filename, bits=None, num_radius=75, smoothing="laplacian", mode="connectivity", metric="minkowski", use_validation=False):
    return MyOwnDataset(".", raw_filename=filename, bits=bits, num_radius=num_radius, smoothing=smoothing, mode=mode, metric=metric, use_validation=use_validation)  # loader.dataset


load_data =load_data_radius
