from typing import Sequence

import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, DataLoader

from scipy.linalg import toeplitz


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, bits, raw_filename, use_validation=False):
        self.use_validation = use_validation
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
        return [f'data-slp-toeplitz-{filename}-{self.use_validation}-{bits}.pt']

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

            train_set = 2326
            valid_set = 100 if self.use_validation else 0
            test_set = df_whole.shape[0] - valid_set

            df_xtrain = df_whole.iloc[0:train_set, 1:]
            df_xtrain = np.array([toeplitz(x) for x in zip(df_xtrain.values)])
            df_ytrain = df_whole['value'][0:train_set]
            df_ytrain = df_ytrain - df_ytrain.mean()

            df_xtest = df_whole.iloc[train_set:test_set, 1:]
            df_xtest = np.array([toeplitz(x) for x in zip(df_xtest.values)])
            df_ytest = df_whole['value'][train_set:test_set]
            df_ytest = df_ytest - df_ytest.mean()
            x_train = torch.from_numpy(df_xtrain).float()
            y_train = torch.tensor([[n] for n in df_ytrain.values], dtype=torch.float)
            x_test = torch.from_numpy(df_xtest).float()
            y_test = torch.tensor([[n] for n in df_ytest.values], dtype=torch.float)
        
            #edge_index = torch.tensor(knn_dist_graph_train.nonzero(), dtype=torch.long)
            data = Data(x=x_train, y=y_train)
            
            #edge_index = torch.tensor(knn_dist_graph_test.nonzero(), dtype=torch.long)
            test_data = Data(x=x_test, y=y_test)
            data.test = test_data
            if self.use_validation:
                data.valid = {"x": torch.tensor(df_whole.iloc[test_set:, 1:].values.tolist(), dtype=torch.float), "y": torch.tensor([[y] for y in df_whole['value'][test_set:].values], dtype=torch.float)}

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_slp_toeplitz(filename, bits=None, use_dataset=False, use_validation=False, batch_size=0, **_):
    return MyOwnDataset(".", bits=bits, raw_filename=filename, use_validation=use_validation)  # loader.dataset


load_data = load_data_slp_toeplitz
