import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

from gnn.loaders.util import split_dataset_graph

os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
from os import path
import pathlib
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset

import time

from gnn.loaders.load import load_data

import numpy as np
import networkx as nx

# from res.plot_lib import set_default
import matplotlib.pyplot as plt

# set_default(figsize=(3, 3), dpi=150)


def add_features(graph, data):
    graph.ndata["feat"] = data.x
    graph.edata["feat"] = torch.ones(graph.number_of_edges(), 1)

    return graph


def get_edges(dataset):
    edges_raw = (
        dataset.edge_index[0][0]
        if isinstance(dataset.edge_index, (tuple, list))
        else dataset.edge_index
    )
    edges_raw = edges_raw.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    return edges


def create_data(loader, filename, **kwargs):
    # filename = str(pathlib.Path("csv-data") / filename)
    dataset = loader(filename, **kwargs)
    print("dataset ==", dataset)

    if len(dataset) > 1:
        data = [x for x in dataset]  # .to(device)
    else:
        data = dataset[0]  # .to(device)

    return data, dataset.num_features


def draw(g, title):
    plt.figure()
    nx.draw(g.to_networkx(), with_labels=True, node_color="skyblue", edge_color="white")
    plt.gcf().set_facecolor("k")
    plt.title(title)


# graph_type = (
#     'cycle',
#     'star',
#     'wheel',
#     'lollipop',
#     'hypercube',
#     'grid',
#     'clique',
#     'circular ladder',
# )
#
# for graph, label in MiniGCDataset(8, 10, 20):
#     draw(graph, f'Class: {label}, {graph_type[label]} graph')
#     plt.show()
#
# Printd


# create artifical data feature (= in degree) for each node
def create_artificial_features(dataset):
    for graph, _ in dataset:
        graph.ndata["feat"] = graph.in_degrees().view(-1, 1).float()

        graph.edata["feat"] = torch.ones(graph.number_of_edges(), 1)

    return dataset


class GatedGCN_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):

        Bx_j = edges.src["BX"]
        # e_j = Ce_j + Dxj + Ex
        e_j = edges.data["CE"] + edges.src["DX"] + edges.dst["EX"]
        edges.data["E"] = e_j
        return {"Bx_j": Bx_j, "e_j": e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data["AX"]
        Bx_j = nodes.mailbox["Bx_j"]
        e_j = nodes.mailbox["e_j"]
        # sigma_j = σ(e_j)
        σ_j = torch.sigmoid(e_j)
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(σ_j * Bx_j, dim=1) / torch.sum(σ_j, dim=1)
        return {"H": h}

    def forward(self, g, X, E_X, snorm_n, snorm_e):

        g.ndata["H"] = X
        g.ndata["AX"] = self.A(X)
        g.ndata["BX"] = self.B(X)
        g.ndata["DX"] = self.D(X)
        g.ndata["EX"] = self.E(X)
        g.edata["E"] = E_X
        g.edata["CE"] = self.C(E_X)

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata["H"]  # result of graph convolution
        E = g.edata["E"]  # result of graph convolution

        H *= snorm_n  # normalize activation w.r.t. graph node size
        E *= snorm_e  # normalize activation w.r.t. graph edge size

        H = self.bn_node_h(H)  # batch normalization
        E = self.bn_node_e(E)  # batch normalization

        H = torch.relu(H)  # non-linear activation
        E = torch.relu(E)  # non-linear activation

        H = X + H  # residual connection
        # edge index
        E = E_X + E  # residual connection

        return H, E


class MLP_layer(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb of hidden layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim, input_dim) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, L):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList(
            [GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)]
        )
        self.MLP_layer = MLP_layer(hidden_dim, output_dim)  # try taking this out

    def forward(self, g, X, E, snorm_n, snorm_e):

        # input embedding
        H = self.embedding_h(X)
        E = self.embedding_e(E)

        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            H, E = GGCN_layer(g, H, E, snorm_n, snorm_e)

        # MLP classifier
        g.ndata["H"] = H
        y = dgl.mean_nodes(g, "H")
        y = self.MLP_layer(y)

        return y


#
# # instantiate network
# model = GatedGCN(input_dim=1, hidden_dim=100, output_dim=8, L=2)
# # print(model)


def get_datalen(filename):
    with open(filename) as fp:
        return fp.read().count("\n")


def create_graphs(filename, neighbours=3, split=0.8):
    filename = str(pathlib.Path("csv-data") / filename)
    data_len = get_datalen(filename)
    raw_data = create_data(
        load_data,
        filename,
        num_neighbours=neighbours,
        batches=data_len-neighbours-1,
        split=data_len-neighbours-1,
        split_algorithm=split_dataset_graph,
        split_algorithm_params={"allow_duplicates": True}
    )

    raw_data, num_features = raw_data
    graphs = []
    values = []
    for datum in raw_data:
        values.append(datum.y[0])
        edges = get_edges(datum)

        g = nx.Graph()
        g.add_edges_from(edges)

        graph = dgl.from_networkx(g)
        graph = add_features(graph, datum)
        graphs.append(graph)

    split = int(data_len*split) if split < 1 else split

    train_g = graphs[:split]
    train_values = torch.Tensor(values[:split])

    test_g = graphs[:split]
    test_values = torch.Tensor(values[split:])

    return list(zip(train_g, train_values)), list(zip(test_g, test_values)), num_features


def collate(samples):

    graphs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)

    labels = torch.tensor(labels)
    sizes_n = [g.number_of_nodes() for g in graphs] # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
    sizes_e = [graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, labels, snorm_n, snorm_e


def train(model, optimizer, data_loader, loss):
    model.train()
    epoch_loss = 0
    nb_data = 0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(
        data_loader
    ):
        batch_X = batch_graphs.ndata["feat"]
        batch_E = batch_graphs.edata["feat"]

        batch_scores = model(
            batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e
        )

        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        epoch_loss += J.detach().item()
        nb_data += batch_labels.size(0)

    epoch_loss /= iter + 1

    return epoch_loss


def evaluate(model, optimizer, data_loader, loss):

    model.eval()
    epoch_test_loss = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (
            batch_graphs,
            batch_labels,
            batch_snorm_n,
            batch_snorm_e,
        ) in enumerate(data_loader):
            batch_X = batch_graphs.ndata["feat"]
            batch_E = batch_graphs.edata["feat"]

            batch_scores = model(
                batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e
            )
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            nb_data += batch_labels.size(0)

        epoch_test_loss /= iter + 1

    return epoch_test_loss


def main(filename, epochs):
    train_data, test_data, num_features = create_graphs(filename)

    train_loader = DataLoader(train_data, batch_size=50, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=50, shuffle=False, collate_fn=collate)

    model = GatedGCN(input_dim=num_features, hidden_dim=100, output_dim=1, L=4)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_loader, loss)
        test_loss = evaluate(model, optimizer, test_loader, loss)

        print(f"Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")


if __name__ == "__main__":
    main("MiceBL.csv", 100)
