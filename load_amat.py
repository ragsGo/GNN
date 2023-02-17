import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from scipy import sparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_networkx


def cov_2_corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    print(A.T / d)
    A = ((A.T / d).T) / d
    # A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A


def A_mat(df):

    s = df.father.values.flatten()
    d = df.mum.values.flatten()
    if (len(s) != len(d)):
        raise TypeError("size of the father vector and mum vector are different!")

    n = len(df)
    N = n
    A = [[0] * N for _ in range(N)]
    # print (len(A))

    # set sires and dams
    s = (s == 0) * (N) + s
    d = (d == 0) * N + d

    # print(s)
    for i in range(n):

        # equation for diagonals

        # print(A[5][5]/2)

        # np.fill_diagonal(A, 1 + A[s[i]][d[i]]/2)
        A[i][i] = 1 + A[s[i]][d[i]] / 2

        for j in range(i + 1, n):  # only do half of the matrix (symmetric)
            if (j > n):
                break
            A[i][j] = (A[i][s[j]] + A[i][d[j]]) / 2  # half relationship to parents
            A[j][i] = A[i][j]  # symmetric matrix, so copy to other off-diag
    return A


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, raw_filename, num_neighbours=10, smoothing="laplacian", mode="distance", use_validation=False):
        self.use_validation = use_validation
        self.num_neighbours = num_neighbours
        self.smoothing = smoothing
        self.mode = mode
        self.raw_filename = raw_filename
        super(MyOwnDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_filename]

    @property
    def processed_file_names(self):
        filename = self.raw_filename.replace("/", ":")
        return [f'amat-data-{filename}-{self.num_neighbours}-{self.smoothing}-{self.mode}.pt']

    def download(self):
        ...

    def process(self):
        data_list = []

        filename = "Pedigree_Data.csv"
        df = pd.read_csv(filename)
        df = df.drop(df.index[(df.father.eq(0) | df.mum.eq(0))])
        train_df = df[:100]
        A = A_mat(train_df)

        x_train = torch.tensor(np.array(A), dtype=torch.float)
        # ytrain = torch.tensor([n for n in y ], dtype=torch.float)
        data = from_networkx(nx.from_numpy_array(np.array(A)))
        data.x = x_train
        data.y = torch.tensor([[n] for n in train_df['value'].values], dtype=torch.float)

        test_df = df[100:200]
        t_A = A_mat(test_df)

        x_test = torch.tensor(np.array(t_A), dtype=torch.float)
        data.test = from_networkx(nx.from_numpy_array(np.array(t_A)))
        data.test.x = x_test
        data.test.y = torch.tensor([[n] for n in test_df['value'].values], dtype=torch.float)

        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_data_amat(filename, num_neighbours=10, smoothing="laplacian", mode="distance", use_validation=False):
    return MyOwnDataset(".", raw_filename=filename, num_neighbours=num_neighbours, smoothing=smoothing, mode=mode, use_validation=use_validation)  # loader.dataset


load_data = load_data_amat