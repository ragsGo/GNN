import pandas as pd
import numpy as np

import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


def create_parent_edges(filename, subset):
    df = pd.read_csv(filename)

    df = df[["mum", "father"]].iloc[subset]

    df.index += 1
    edges = np.zeros((df.shape[0], df.shape[0]))

    for x in range(df.shape[0]):
        for y in range(x+1, df.shape[0]):
            edges[x, y] += 0.5 if x == df.iloc[y]["father"] else 0
            edges[x, y] += 0.5 if x == df.iloc[y]["mum"] else 0
            edges[y, x] = edges[x, y]
    return edges


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, num_neighbours=1, smoothing="laplacian", mode="connectivity",
                 use_validation=False):
        self.use_validation = use_validation
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
        return [f'data-parents-{filename}-{self.use_validation}-{bits}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []
        for filename in self.raw_file_names:

            with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
            value_columns = [str((i + 1)) for i in range(column_count - 1)]
            labels = ["value"] + value_columns
            df_whole = pd.read_csv(filename, names=labels)
            if self.bits is not None:
                df_whole = df_whole[["value"] + self.bits]

            df_whole['value'] = df_whole['value'] - df_whole['value'].mean()

            df_new = pd.DataFrame()
            df_new["value"] = df_whole["value"]
            labels = []
            for i in range(1, len(df_whole.columns)):
                df_new[i] = (df_whole[str(i)] == 1).astype(int)
                labels.append(i)
            for i in range(1, len(df_whole.columns)):
                df_new[i+len(df_whole.columns)-1] = (df_whole[str(i)] == 2).astype(int)
                labels.append(i+len(df_whole.columns)-1)

            df_whole = df_new
            train_set = 2326
            valid_set = 100 if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole.iloc[0:train_set, 1:]
            df_ytrain = df_whole['value'][0:train_set]

            df_xtest = df_whole.iloc[train_set:test_set, 1:]
            df_ytest = df_whole['value'][train_set:test_set]

            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)
            x_test = torch.tensor(df_xtest.values.tolist(), dtype=torch.float)
            y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)

            parent_train = create_parent_edges("Pedigree_Data.csv", slice(0, train_set))
            parent_test = create_parent_edges("Pedigree_Data.csv", slice(train_set, test_set))

            edge_index = torch.tensor(parent_train, dtype=torch.long)
            data = Data(x=x_train, y=y_train, edge_index=edge_index)
            edge_index = torch.tensor(parent_test, dtype=torch.long)
            test_data = Data(x=x_test, y=y_test, edge_index=edge_index)

            data.test = test_data
            if self.use_validation:
                data.valid = {"x": torch.tensor(df_whole.iloc[test_set:, 1:].values.tolist(), dtype=torch.float),
                              "y": torch.tensor([[y] for y in df_whole['value'][test_set:].values], dtype=torch.float),
                              "edge_index": create_parent_edges("Pedigree_Data.csv", slice(test_set, df_whole.shape[0]))}

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_hot2_hop(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False):
    return MyOwnDataset(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing,
                        mode=mode, use_validation=use_validation)  # loader.dataset


load_data = load_data_hot2_hop
