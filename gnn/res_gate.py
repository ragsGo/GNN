import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
os.environ['DGLBACKEND'] = 'pytorch'  # tell DGL what backend to use
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
        graph.ndata['feat'] = data.x
        graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)

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

def create_data(loader,filename):
    filename = str(pathlib.Path("csv-data") / filename)
    dataset = loader(filename)
    print('dataset ==', dataset)

    if len(dataset) > 1:
        data = [x for x in dataset]#.to(device)
    else:
        data = dataset[0]#.to(device)

    return data, dataset.num_features



def draw(g, title):
    plt.figure()
    nx.draw(g.to_networkx(), with_labels=True, node_color='skyblue', edge_color='white')
    plt.gcf().set_facecolor('k')
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
    for (graph, _) in dataset:
        graph.ndata['feat'] = graph.in_degrees().view(-1, 1).float()

        graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)

    return dataset



trainset = MiniGCDataset(350, 10, 20)
# print(trainset)
#
# prints
testset = MiniGCDataset(100, 10, 20)

trainset = create_artificial_features(trainset)
# print(trainset)
# prints
testset = create_artificial_features(testset)
print("trainset----",trainset[1])
print("testset==",testset[1])

loader = load_data
dataset = "MiceBL.csv"
data = create_data(loader,dataset)

data, inp_size = data

train_edges = get_edges(data)
test_edges = get_edges(data.test)

g = nx.Graph()
g.add_edges_from(train_edges)

g_test = nx.Graph()
g_test.add_edges_from(test_edges)

train_graph = dgl.from_networkx(g)
train_set = add_features(train_graph, data)

test_graph = dgl.from_networkx(g_test)
test_set = add_features(test_graph, data.test)

print("train_set----",train_set)
print("test_set==",test_set)


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

        Bx_j = edges.src['BX']
        # e_j = Ce_j + Dxj + Ex
        e_j = edges.data['CE'] + edges.src['DX'] + edges.dst['EX']
        edges.data['E'] = e_j
        return {'Bx_j' : Bx_j, 'e_j' : e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        # sigma_j = σ(e_j)
        σ_j = torch.sigmoid(e_j)
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(σ_j * Bx_j, dim=1) / torch.sum(σ_j, dim=1)
        return {'H' : h}

    def forward(self, g, X, E_X, snorm_n, snorm_e):



        g.ndata['H']  = X
        g.ndata['AX'] = self.A(X)
        g.ndata['BX'] = self.B(X)
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X)
        g.edata['E']  = E_X
        g.edata['CE'] = self.C(E_X)

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata['H'] # result of graph convolution
        E = g.edata['E'] # result of graph convolution

        H *= snorm_n # normalize activation w.r.t. graph node size
        E *= snorm_e # normalize activation w.r.t. graph edge size

        H = self.bn_node_h(H) # batch normalization
        E = self.bn_node_e(E) # batch normalization

        H = torch.relu(H) # non-linear activation
        E = torch.relu(E) # non-linear activation

        H = X + H # residual connection
        # edge index
        E = E_X + E # residual connection

        return H, E

class MLP_layer(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): # L = nb of hidden layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim, input_dim) for l in range(L)
        ]
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
        self.GatedGCN_layers = nn.ModuleList([
            GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)
        ])
        self.MLP_layer = MLP_layer(hidden_dim, output_dim) #try taking this out

    def forward(self, g, X, E, snorm_n, snorm_e):

        # input embedding
        H = self.embedding_h(X)
        E = self.embedding_e(E)

        # graph convnet layers
        for GGCN_layer in self.GatedGCN_layers:
            H, E = GGCN_layer(g, H, E, snorm_n, snorm_e)

        # MLP classifier
        g.ndata['H'] = H
        y = dgl.mean_nodes(g, 'H')
        y = self.MLP_layer(y)

        return y
#
# # instantiate network
# model = GatedGCN(input_dim=1, hidden_dim=100, output_dim=8, L=2)
# # print(model)

def collate(samples):
    print("samples ===", samples)

    # graphs, labels = map(list, zip(*samples))  # samples is a list of pairs (graph, label)
    graph = samples
    print(type(graph[0]))
    # labels = torch.tensor(labels)
    sizes_n = graph[0].number_of_nodes#[g.number_of_nodes() for g in graph] # graph sizes
    print (sizes(n))
    snorm_n = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_n]
    snorm_n = torch.cat(snorm_n).sqrt()  # graph size normalization
    sizes_e = graph.number_of_edges  #[graph.number_of_edges() for graph in graphs] # nb of edges
    snorm_e = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes_e]
    snorm_e = torch.cat(snorm_e).sqrt()  # graph size normalization
    batched_graph = dgl.batch(graphs)  # batch graphs
    return batched_graph, labels, snorm_n, snorm_e

# Compute accuracy

def accuracy(logits, targets):
    preds = logits.detach().argmax(dim=1)
    acc = (preds==targets).sum().item()
    return acc


# Define DataLoader and get first graph batch
#
# train_loader = DataLoader(trainset, batch_size=10, shuffle=True, collate_fn=collate)
# batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e = next(iter(train_loader))
# batch_X = batch_graphs.ndata['feat']
# batch_E = batch_graphs.edata['feat']
# print(batch_graphs)
# print(batch_labels)
# prints
#
# # Checking some sizes
#
# print(f'batch_graphs:', batch_graphs)
# print(f'batch_labels:', batch_labels)
# print('batch_X size:', batch_X.size())
# print('batch_E size:', batch_E.size())
#
# batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
# print(batch_scores.size())
#
# batch_labels = batch_labels
# print(f'accuracy: {accuracy(batch_scores, batch_labels)}')
#
# # Loss
# J = nn.CrossEntropyLoss()(batch_scores, batch_labels)
#
# # Backward pass
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer.zero_grad()
# J.backward()
# optimizer.step()


def train(model, data_loader, loss):
    print(list(data_loader))
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_X = batch_graphs.ndata['feat']
        batch_E = batch_graphs.edata['feat']

        batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        epoch_loss += J.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)

    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc


def evaluate(model, data_loader, loss):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_X = batch_graphs.ndata['feat']
            batch_E = batch_graphs.edata['feat']

            batch_scores = model(batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e)
            J = loss(batch_scores, batch_labels)

            epoch_test_loss += J.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)

        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc

train_loader = DataLoader(train_set, batch_size=50, shuffle=False, collate_fn=collate)
# print('ksksksk')
# train_loader = DataLoader(train_set, batch_size=50, shuffle=False, collate_fn=collate)

# test_loader = DataLoader(test_set, batch_size=50, shuffle=False, collate_fn=collate)

# Create model
model = GatedGCN(input_dim=1, hidden_dim=100, output_dim=8, L=4)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epoch_train_losses = []
epoch_test_losses = []
epoch_train_accs = []
epoch_test_accs = []

for epoch in range(40):

    start = time.time()
    train_loss, train_acc = train(model, train_loader, loss)
    test_loss, test_acc = evaluate(model, test_loader, loss)

    print(f'Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}')
    print(f'train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}')
