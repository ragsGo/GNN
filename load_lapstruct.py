import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from torch_geometric.utils import from_networkx

from lapstruct import lapstruct

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=10,  smoothing="laplacian", mode="distance", num_vars=2, use_validation=False):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.num_vars = num_vars
        self.bits = bits
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
        return [f'data-{filename}-{self.mode}-{self.num_neighbours}-{self.smoothing}-{self.num_vars}-{bits}.pt']

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

            n_neighbors = self.num_neighbours
            G = df_whole
            G_train = G.oc[:2326,:]
            G_test = G.loc[2326:,:]
            
            
            y_train = torch.tensor([n for n in G_train.iloc[:,0]], dtype=torch.float)
            G_train = G_train.iloc[ :, 1:]
            l = lapstruct(G_train.values)
            x_train = torch.tensor([n for n in l[:,-self.num_vars:].tolist()])
            
            
            y_test = torch.tensor([n for n in G_test.iloc[:,0]], dtype=torch.float)
            G_test = G_test.iloc[ :, 1:]
            l_test = lapstruct(G_test.values)
            x_test = torch.tensor([n for n in l_test[:,-self.num_vars:].tolist()], dtype=torch.float)
            
                       
            knn_dist_graph_train = kneighbors_graph(X=x_train,
                                              n_neighbors=n_neighbors,
                                              mode=self.mode,
                                              n_jobs=6)
            knn_dist_graph_test = kneighbors_graph(X=x_test,
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
            #data.train_mask = torch.tensor(mask, dtype=torch.bool)
            #data.test_mask = torch.tensor([0 if n else 1 for n in mask], dtype=torch.bool)
            data.test = test_data
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_lapstruct(filename, bits=None, num_neighbours=10,  smoothing="laplacian", mode="distance", num_vars=2, use_validation=False):

    return MyOwnDataset(".",  bits=None, raw_filename=filename, num_neighbours=num_neighbours,  smoothing=smoothing, mode=mode, num_vars=num_vars, use_validation=use_validation)  # loader.dataset


load_data = load_data_lapstruct
