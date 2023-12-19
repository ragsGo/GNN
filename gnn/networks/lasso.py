from lassonet import  LassoNet

from torch.nn import ReLU, Linear, Dropout
from torch_geometric.nn import GCNConv


from gnn.networks.sequential import Sequential


def create_network_lasso_no_conv_relu_dropout(inp_size, out_size, internal_size=-1, internal_dropout=None, groups=None, dropout=0.5, *_, **__):
    if internal_size <= 0:
        internal_sizeinternal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (LassoNet(inp_size, internal_size, internal_size, groups=groups, dropout=internal_dropout), 'x -> x'),
        (GCNConv(internal_size, internal_size), 'x, edge_index, edge_weights -> x'),
        ReLU(),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model
