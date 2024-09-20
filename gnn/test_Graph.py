import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb
import time
import numpy as np
import pickle
from os import path
import os
import pathlib

import dgl
import torch as th
from gnn.loaders.load import load_data
import numpy as np
import networkx as nx
# from res.plot_lib import set_default
import matplotlib.pyplot as plt


def add_features(graph, data):
        print(graph)
        # print(data.x)
        graph.ndata['feat'] = data.x
        print(graph.ndata['feat'])
        graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)
        print(graph.edata['feat'])

        prints
        return dataset

def get_edges(dataset):
    edges_raw = (
        dataset.edge_index[0][0]
        if isinstance(dataset.edge_index, (tuple, list))
        else dataset.edge_index
    )
    edges_raw = edges_raw.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    return edges

def draw(g, title):
    plt.figure()
    nx.draw(g.to_networkx(), with_labels=True, node_color='skyblue', edge_color='white')
    plt.gcf().set_facecolor('k')
    plt.title(title)

def create_data(loader,filename):
    filename = str(pathlib.Path("csv-data") / filename)
    dataset = loader(filename)
    print('dataset ==', dataset)

    if len(dataset) > 1:
        data = [x for x in dataset]#.to(device)
    else:
        data = dataset[0]#.to(device)

    return data, dataset.num_features

loader = load_data
dataset = "MiceBL.csv"
data = create_data(loader,dataset)

data, inp_size = data

edges = get_edges(data)

g = nx.Graph()
g.add_edges_from(edges)

d_graph = dgl.from_networkx(g)
d_graph = add_features(d_graph, data)
# draw(d_graph, "dgl  graph")
# plt.show()

print(d_graph)
