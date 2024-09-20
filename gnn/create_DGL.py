import dgl

import torch as th

import scipy.sparse as sp

spmat = sp.rand(100, 100, density=0.05) # 5% nonzero entries

# print(spmat)

dgl.from_scipy(spmat)                   # from SciPy


import networkx as nx

nx_g = nx.path_graph(5) # a chain 0-1-2-3-4
print(nx_g)
dgl.from_networkx(nx_g) # from networkx
