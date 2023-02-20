#
# from gnn.loaders.load import load_data
# from gnn.main import create_data


import random
import networkx as nx


def forest_fire(edges, size: int, traversed: set = None, **_):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    nodes = list(graph.nodes())
    sampled_graph = nx.Graph()
    if traversed is None:
        traversed = set()
    random_node = random.sample(nodes, 1)[0]
    q = set()   # q = set contains the distinct values
    q.add(random_node)
    while len(sampled_graph.nodes()) < size:
        if len(q) > 0:
            initial_node = q.pop()
            if initial_node not in traversed:
                traversed.add(initial_node)
                neighbours = list(graph.neighbors(initial_node))
                np = random.randint(1, len(neighbours))
                for x in neighbours[:np]:
                    if len(sampled_graph.nodes()) < size:
                        sampled_graph.add_edge(initial_node, x)
                        q.add(x)
                    else:
                        break
            else:
                continue
        else:
            random_node = random.sample(list(set(nodes) - traversed), 1)[0]
            q.add(random_node)
    return list(sampled_graph.nodes())

#
# dataset = create_data(load_data, params={'filename': "csv-data/WHEAT_combined.csv"}, plot=False)
# data, inp_size = dataset
# edges_raw = data.edge_index[0][0] if isinstance(data.edge_index, (tuple, list)) else data.edge_index
# edges_raw = edges_raw.numpy()
# edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
#
# sample4 = forest_fire(edges, 100)  # graph, number of nodes to sample
# print("Forest Fire Sampling")
# print("Number of nodes sampled=", len(sample4.nodes()))
# print(sample4.nodes())
# print("Number of edges sampled=", len(sample4.edges()))
