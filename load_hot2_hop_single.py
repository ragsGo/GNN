import pandas as pd
import numpy as np

import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


def create_parent_edges(filename):
    df = pd.read_csv(filename)

    df = df[["mum", "father"]]

    edges = np.zeros((df.shape[0], df.shape[0]))

    for x in range(df.shape[0]):
        f = int(df.iloc[x]["father"]) - 1
        m = int(df.iloc[x]["mum"]) - 1
        edges[x, f] += 0.5
        edges[x, m] += 0.5
        edges[f, x] = edges[x, f]
        edges[m, x] = edges[x, m]
    return edges


class Test:
    def __init__(self, rng):
        self.range = rng


class Validation:
    def __init__(self, rng):
        self.range = rng


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
        return [f'data-hot2-hop-single-{filename}-{self.use_validation}-{bits}.pt']

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

            df_xtrain = df_whole.iloc[:, 1:]
            df_ytrain = df_whole['value']

            x_train = torch.tensor(df_xtrain.values.tolist(), dtype=torch.float)
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)

            parent_train = create_parent_edges("Pedigree_Data.csv")

            edge_index = torch.tensor(parent_train, dtype=torch.long)
            data = Data(x=x_train, y=y_train, edge_index=edge_index)
            data.range = range(0, train_set)
            test_data = Test(range(train_set, test_set))
            data.test = test_data
            if self.use_validation:
                data.valid = Validation(range(test_set, df_whole.shape[0]))

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_hot2_hop_single(filename, bits=None, num_neighbours=1, smoothing="laplacian", mode="connectivity", use_validation=False):
    return MyOwnDataset(".", bits=bits, raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing,
                        mode=mode, use_validation=use_validation)  # loader.dataset


load_data = load_data_hot2_hop_single
