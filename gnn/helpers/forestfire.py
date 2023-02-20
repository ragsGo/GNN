import os
import pathlib
import pickle
import shutil
import time
import random
import networkx as nx
import scipy
from bayes_opt import BayesianOptimization
import torch
from bayes_opt import SequentialDomainReductionTransformer
from hyperopt import delete, save, load
from load_ensembles2 import load_data as load_data_ensemble2
from load import load_data
from main import create_data, l1_regularize, all_the_things
from networks import create_network_conv_two_diff, create_network_conv1D, \
    create_network_two_diff, create_network_no_conv, create_network_no_conv, create_network_no_conv_dropout, \
    create_network_two_no_conv_dropout, create_network_two_no_conv_relu_dropout, create_network_no_conv_relu_dropout
from load_two import load_data as load_data_two
from load_scaled import load_data as load_data_scaled
from load_batches import load_data as load_data_batches

from load_hot2 import load_data as load_data_hot2
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx


class ForestFire():
    def __init__(self):
        self.G1 = nx.Graph()

    def forestfire(self, G, size):
        list_nodes = list(G.nodes())
        # print(len(G))
        dictt = set()
        random_node = random.sample(set(list_nodes), 1)[0]
        print("random node=", random_node)
        q = set()   # q = set contains the distinct values
        q.add(random_node)
        print('q val ==',q)
        while(len(self.G1.nodes()) < size):
            print('q===', q)
            print('len(q)===', len(q))
            print('len val--', len(q) > 0)
            print(dictt)
            if(len(q) > 0):
                initial_node = q.pop()
                print('here1')
                if(initial_node not in dictt):
                    print(initial_node)
                    dictt.add(initial_node)
                    neighbours = list(G.neighbors(initial_node))
                    # print(list(G.neighbors(initial_node)))
                    np = random.randint(1, len(neighbours))
                    print("np===", dictt)
                    print("neighbours[:np]===", neighbours[:np])
                    for x in neighbours[:np]:
                        if(len(self.G1.nodes()) < size):
                            self.G1.add_edge(initial_node, x)
                            q.add(x)
                            print('here == ',x)
                            print(self.G1)
                        else:
                            break
                else:
                    continue
            else:
                random_node = random.sample(set(list_nodes) and dictt, 1)[0]
                print("random --",random_node)
                q.add(random_node)
                print('if qval --', q)
        q.clear()
        print(self.G1)
        return self.G1


dataset = create_data(load_data, params={'filename': "csv-data/WHEAT_combined.csv"})
data, inp_size = dataset
edges_raw = data.edge_index[0][0] if isinstance(data.edge_index, (tuple, list)) else data.edge_index
edges_raw = edges_raw.numpy()
edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
# edges = [(x, y) for x, y in edges if x < 250 and y < 250]
G = nx.Graph()
# G.add_nodes_from(list(range(np.max(edges_raw))))
G.add_edges_from(edges)
print (G)

object4 = ForestFire()
sample4 = object4.forestfire(G,3) # graph, number of nodes to sample
print("Forest Fire Sampling")
print("Number of nodes sampled=",len(sample4.nodes()))
print(sample4.nodes())
print("Number of edges sampled=",len(sample4.edges()))
