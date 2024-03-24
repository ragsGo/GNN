import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import os

from gnn.loaders.util import split_dataset_graph

import pathlib
from gnn.loaders.load import load_data

import networkx as nx


os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl


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


class GatedGCNLayer(nn.Module):

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


class MLPLayer(nn.Module):

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
            [GatedGCNLayer(hidden_dim, hidden_dim) for _ in range(L)]
        )
        self.MLP_layer = MLPLayer(hidden_dim, output_dim)  # try taking this out

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
        batches=data_len - neighbours - 1,
        split=data_len - neighbours - 1,
        split_algorithm=split_dataset_graph,
        split_algorithm_params={"allow_duplicates": True},
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

    split = int(data_len * split) if split < 1 else split

    train_g = graphs[:split]
    train_values = torch.Tensor(values[:split])

    test_g = graphs[:split]
    test_values = torch.Tensor(values[split:])

    return (
        list(zip(train_g, train_values)),
        list(zip(test_g, test_values)),
        num_features,
    )


def collate(samples):

    graphs, labels = map(
        list, zip(*samples)
    )  # samples is a list of pairs (graph, label)

    labels = torch.tensor(labels)
    sizes_n = [g.number_of_nodes() for g in graphs]  # graph sizes
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
    sizes_e = [graph.number_of_edges() for graph in graphs]  # nb of edges
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


def reverse(
    model,
    optim,
    inp_size,
    data_loader,
    loss_func=torch.nn.MSELoss(),
    num_steps=250,
    select_size=(100, 100),
    sort_labels=True,
    plot=False,
    save_name="save.png",
):
    model.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        batch_graphs,
        batch_labels,
        batch_snorm_n,
        batch_snorm_e,
    ) = collate(data_loader)
    inp_size = batch_graphs.ndata["feat"].shape
    x = torch.rand(inp_size, requires_grad=True, device=device)
    optim.add_param_group({"params": x})

    for epoch in range(num_steps):
        batch_E = batch_graphs.edata["feat"]
        batch_X = x

        batch_scores = model(
            batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e
        )

        J = loss_func(batch_scores, batch_labels)
        optim.zero_grad()
        J.backward()
        optim.step()

        print("Reverse epoch: {:03d}, Loss: {:.10f}".format(epoch, J))

    if isinstance(select_size, int):
        select_size = (select_size, select_size)

    range_size = int(inp_size[0] / select_size[1])
    selections = list(range(0, inp_size[0], range_size if range_size > 0 else 1))
    values = x.cpu().detach().numpy()[selections, : select_size[0]]

    if plot:
        y_labels = batch_labels.detach().squeeze().numpy()
        if sort_labels:
            val_labels = zip(values, y_labels)
            values, y_labels = zip(*sorted(list(val_labels), key=lambda k: k[1]))
        plt.figure(figsize=(12, 50))
        plt.imshow(values)
        plt.xlabel(r"Loc", fontsize=12)
        plt.ylabel(r"Value", fontsize=12)
        plt.yticks(np.arange(0, len(selections)), y_labels[selections])
        plt.colorbar()
        plt.savefig(save_name)

    return x


def main(filename, epochs):
    train_data, test_data, num_features = create_graphs(filename)

    train_loader = DataLoader(
        train_data, batch_size=50, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_data, batch_size=50, shuffle=False, collate_fn=collate
    )

    model = GatedGCN(input_dim=num_features, hidden_dim=100, output_dim=1, L=4)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        train_loss = train(model, optimizer, train_loader, loss)
        test_loss = evaluate(model, optimizer, test_loader, loss)

        print(
            f"Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}"
        )

    x = (
        reverse(
            model,
            torch.optim.Adam(model.parameters(), lr=0.004),
            torch.Size([num_features, len(train_data)]),
            train_data,
        )
        .cpu()
        .detach()
        .numpy()
    )
    print(x)


if __name__ == "__main__":
    main("MiceBL.csv", 100)
