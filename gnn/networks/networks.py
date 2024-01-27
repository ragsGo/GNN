import numpy as np
import torch
from torch import nn
from torch.nn import ReLU, Conv1d, MaxPool1d, Flatten, Linear, Dropout, Conv2d, MaxPool2d, Identity
from torch_geometric.nn import GCNConv, ResGatedGraphConv
from torch_geometric.nn.dense import DenseGCNConv
from torch.nn import functional as F

from .cov_gnn import CovGnn
from .multi_gnn import GCNAISUMMER
from .sequential import Sequential


from .two import TwoGNN
from .two_diff import TwoDiffGNN


# Taken from https://discuss.pytorch.org/t/what-is-reshape-layer-in-pytorch/1110/6
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Embed(nn.Module):
    def __init__(self,inp_size,embedding_dim):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(inp_size, embedding_dim)

    def forward(self, x):
        print('embedding ===', self.embedding)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.tensor(x).to(device).long()
        print(x1.shape)
        embedded = self.embedding(x1)
        print(embedded.shape)
        #embedded = embedded.unsqueeze(1)
        #embedded = embedded.squeeze(0)
        print('now===',embedded.shape)
        return(embedded)


class Unsqeeze(nn.Module):
    def __init__(self, dim):
        super(Unsqeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(2, 1, 0)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Ones(nn.Module):
    def __init__(self, size):
        super(Ones, self).__init__()
        self.size = size
        self.weights = torch.tensor(np.ones(self.size)).view(1, 1, size).float()

    def forward(self, x):
        return F.conv1d(x, self.weights)


class Nop(nn.Module):
    def __init__(self):
        super(Nop, self).__init__()

    def forward(self, *args):
        return args if len(args) > 1 else args[0]


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, *args):
        #print(*args)
        print(*[x.shape for x in args])
        return args if len(args) > 1 else args[0]


class Ensemble(nn.Module):
    def __init__(self, models, in_size=1, out_size=1, aggregator=Linear):
        super(Ensemble, self).__init__()
        self.nets = models
        self.aggregator = aggregator(len(models)*in_size, out_size)

    def forward(self, *args):
        results = []
        for net in self.nets:
            result = net(*args)
            results.append(result.view(result.size(0), -1))
        return self.aggregator(torch.cat(tuple(results), dim=1))

    def train(self, **kwargs):
        for net in self.nets:
            net.train()
        self.aggregator.train()

    def eval(self):
        for net in self.nets:
            net.eval()
        self.aggregator.eval()

    def __str__(self):
        return "Ensemble"

    def __repr__(self):
        return str(self)


def create_network_linear(inp_size, out_size, *_, **__):
    model = Sequential('x, edge_index, edge_weights?', [
        (Linear(inp_size, out_size), "x -> x"),
    ])
    return model


def create_network_no_conv_pool(inp_size, out_size, conv_kernel_size=30, pool_size=2):
    out_conv = inp_size

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, inp_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, inp_size),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),  # put l1 regularization
    ])
    return model


def create_network_no_conv_relu(inp_size, out_size, internal_size=-1):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        ReLU(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_no_conv_relu_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1,
                                        dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        ReLU(),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_no_conv_relu_dropout(
        inp_size,
        out_size,
        internal_size=-1,
        dropout=0.5,
        add_self_loops=True,
        **_
):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size), 'x, edge_index, edge_weights -> x'),
        ReLU(),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_no_conv_relu(inp_size, out_size, internal_size=-1,
                                        add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size), 'x, edge_index, edge_weights -> x'),
        ReLU(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_dense_no_conv_dropout(
        inp_size,
        out_size,
        internal_size=-1,
        dropout=0.5,
        **_
):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (DenseGCNConv(inp_size, internal_size), 'x, edge_index -> x'),
        (DenseGCNConv(internal_size, internal_size), 'x, edge_index -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_no_conv_dropout(inp_size, out_size, internal_size=-1, dropout=0.5, add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_no_conv_dropout(inp_size, out_size, internal_size=-1, dropout=0.5, add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_ensemble_creator(*models, out_size=1, aggregator=Linear):
    def creator(*args, **kwargs):
        modules = []
        for m in models:
            new_module = m(*args, **kwargs)
            modules.append(new_module)
        return Sequential('x, edge_index, edge_weights?',
              [(
                  Ensemble(modules, out_size=out_size, aggregator=aggregator),
                  'x, edge_index, edge_weights -> x'
              )]
        )

    return creator


def create_network_test(inp_size, out_size, internal_size=-1, dropout=0.5, add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (Linear(inp_size, internal_size), 'x -> x'),
        (GCNConv(internal_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_graph_conv_two_dropout(
        inp_size,
        out_size,
        conv_kernel_size=30,
        pool_size=2,
        internal_size=-1,
        dropout=0.5
):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Flatten(),
        (GCNConv(out_conv, out_conv), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_conv, out_size),
    ])
    return model


def create_network_no_conv(inp_size, out_size, internal_size=-1, add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_no_conv(inp_size, out_size, internal_size=-1, add_self_loops=True, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size, add_self_loops=add_self_loops), 'x, edge_index, edge_weights -> x'),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_k_hop(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, radius=2):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNAISUMMER(inp_size, internal_size, radius), 'x, edge_index -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_k_hop_no_conv(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, radius=2):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNAISUMMER(inp_size, internal_size, radius), 'x, edge_index -> x'),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_graph_conv_two(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        (GCNConv(internal_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_conv_cov(
        inp_size,
        out_size,
        conv_kernel_size=30,
        pool_size=2,
        internal_size=-1,
        max_neighbours=2,
        radius=2
):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (CovGnn(inp_size, internal_size, max_neighbours, radius), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size), *
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_no_conv_cov(inp_size, out_size, internal_size=-1, max_neighbours=2, radius=2):
    if internal_size <= 0:
        internal_size = inp_size
    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (CovGnn(inp_size, internal_size, max_neighbours, radius), 'x, edge_index, edge_weights -> x'),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_conv(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_2hidden_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_pool = internal_size
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Linear(internal_size, internal_size),
        Linear(internal_size, out_pool),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_gated_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, num_gates=1, num_gnn=1, num_conv=0, dropout=0.5, *_, **__):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = internal_size if num_gates+num_gnn > 0 else inp_size
    convs: list = [Reshape(1, internal_size)]*min(num_conv, 1)
    for _ in range(num_conv):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)
        out_conv = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
        convs.extend([
            Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=pool_size),
        ])

    out_pool = out_conv
    model = Sequential('x, edge_index, edge_weights?',
        [(ResGatedGraphConv(inp_size, internal_size), 'x, edge_index -> x')]*min(num_gates, 1) +
        [(ResGatedGraphConv(internal_size, internal_size), 'x, edge_index -> x')]*max(0, num_gates-1) +
        [(GCNConv(internal_size if num_gates > 0 else inp_size, internal_size), 'x, edge_index, edge_weights -> x')]*min(num_gnn, 1) +
        [(GCNConv(internal_size, internal_size), 'x, edge_index, edge_weights -> x')]*max(0, num_gnn-1) +
        convs +
        [Dropout(dropout), Linear(out_pool, out_size),]
    )
    return model


def create_network_conv_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_2conv_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = internal_size
    for i in range(2):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_pre_conv_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = inp_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (Reshape(1, inp_size), 'x -> x'),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        # Reshape(internal_size),
        (GCNConv(out_pool, out_pool), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_pre_conv2_dropout(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = inp_size
    for i in range(2):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (Reshape(1, inp_size), 'x -> x'),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        # Reshape(internal_size),
        (GCNConv(out_pool, out_pool), 'x, edge_index, edge_weights -> x'),
        Dropout(dropout),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_conv_two(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (TwoGNN(input_size=inp_size, output_size=internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_conv_two_diff(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, radius=2):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = internal_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (TwoDiffGNN(input_size=inp_size, output_size=internal_size, radius=radius), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_two_diff(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, radius=2):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_pool = internal_size

    model = Sequential('x, edge_index, edge_weights?', [
        (TwoDiffGNN(input_size=inp_size, output_size=internal_size, radius=radius), 'x, edge_index, edge_weights -> x'),
        Flatten(),
        Linear(out_pool, out_size),
    ])

    return model


def create_network_2D_graph(
        inp_size,
        out_size,
        embedding_dim,
        filter_size,
        conv_kernel_size=37,
        pool_size=2,
        internal_size=-1,
        **_
):
    if internal_size <= 0:
        internal_size = inp_size
    print('inp size--', inp_size)
    model = Sequential(
        'x, edge_index, edge_weights?',
        [
            (Reshape(1, inp_size), 'x -> x'),
            Embed(inp_size,embedding_dim),Squeeze(1),Print(),
            Conv2d(
                in_channels=1,
                out_channels=conv_kernel_size,
                stride=1,
                padding=1,
                kernel_size=(filter_size[0], embedding_dim)
            ),
            Conv2d(
                in_channels=1,
                out_channels=conv_kernel_size,
                stride=1,
                padding=1,
                kernel_size=(filter_size[1], embedding_dim)
            ),
            Conv2d(
                in_channels=1,
                out_channels=conv_kernel_size,
                stride=1,
                padding=1,
                kernel_size=(filter_size[2], embedding_dim)
            ),
            Print(),
            ReLU(),
            Print(),
            MaxPool1d(kernel_size=pool_size),
            Linear(1 * conv_kernel_size, out_size)
        ]
    )
    return model


def create_network_no_graph(inp_size, out_size, conv_kernel_size=37, pool_size=2, internal_size=-1, **_):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = inp_size

    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (Reshape(1, inp_size), 'x -> x'),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, bias=False),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(4846, out_size),
    ])
    return model


def create_network_ones_hop(inp_size, out_size, conv_kernel_size=37, internal_size=-1, radius=2, as_double=False, **_):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = inp_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    model = Sequential("x, edge_index, edge_weights?", [
        (Unsqeeze(1), "x -> x"),
        Ones(conv_kernel_size),
        Squeeze(1),
        (GCNAISUMMER(out_conv, internal_size, radius), 'x, edge_index -> x'),
        Flatten(),
        Linear(internal_size, out_size)])
    if as_double:
        model.double()
    return model


def create_network_ones(inp_size, out_size, conv_kernel_size=37, internal_size=-1, as_double=False, **_):
    if internal_size <= 0:
        internal_size = inp_size
    internal_size += internal_size % 2

    out_conv = inp_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    model = Sequential("x, edge_index, edge_weights?", [
        (Unsqeeze(1), "x -> x"),
        Ones(conv_kernel_size),
        Squeeze(1),
        (GCNConv(out_conv, internal_size), 'x, edge_index, edge_weights -> x'),
        Flatten(),
        Linear(internal_size, out_size)])
    if as_double:
        model.double()
    return model


def create_network_conv_1D(inp_size, out_size, conv_kernel_size=4, filters=1, as_double=False, **_):
    out_conv = inp_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / conv_kernel_size + 1)

    model = nn.Sequential(
        Reshape(1, 1, inp_size),
        Print(),
        Conv1d(in_channels=1, out_channels=filters, kernel_size=(1, conv_kernel_size),  padding=0),
        Print(),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(1, conv_kernel_size), padding=0),
        Flatten(),
        Linear(filters * out_conv, out_size)
    )
    if as_double:
        model.double()
    return model


def create_network_conv2D(inp_size, out_size, conv_kernel_size=4, filters=1, as_double=False, **_):

    out_conv = inp_size
    for i in range(1):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / conv_kernel_size + 1)

    model = nn.Sequential(
        Reshape(1, 1, inp_size),
        Conv2d(in_channels=1, out_channels=filters, kernel_size=(1, conv_kernel_size),  padding=0),
        ReLU(inplace=True),
        MaxPool2d(kernel_size=(1, conv_kernel_size), padding=0),
        Flatten(),

        Linear(filters * out_conv, out_size)
    )
    if as_double:
        model.double()
    return model


def create_network_conv1D(inp_size, out_size, conv_kernel_size=None, filters=None, as_double=False, **_):

    print('conv_kernel_size==', conv_kernel_size)
    print('filters==', filters)
    if filters is None:
        filters = [20,25]
    elif isinstance(filters, int):
        filters = [filters, filters+5]
    if conv_kernel_size is None:
        conv_kernel_size = [10, 15]
    elif isinstance(conv_kernel_size, int):
        conv_kernel_size = [conv_kernel_size, conv_kernel_size+5]
    out_conv = inp_size
    for i in range(2):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size[i] - 1) - 1) / 1 + 1)
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size[i] - 1) - 1) / conv_kernel_size[i] + 1)

    out_conv *= filters[-1]
    # out_conv = int(out_conv/2)

    model = nn.Sequential(
        Unsqeeze(1),
        Conv1d(in_channels=1, out_channels=filters[0], kernel_size=conv_kernel_size[0], padding=0),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=conv_kernel_size[0], padding=0),
        Conv1d(in_channels=filters[0], out_channels=filters[1], kernel_size=conv_kernel_size[1], padding=0),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=conv_kernel_size[-1], padding=0),
        Flatten(),
        Linear(out_conv, out_size),
        # Squeeze(1),
    )
    if as_double:
        model.double()
    return model


def create_network_2conv(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(2):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=pool_size),
        Flatten(),
        Linear(out_pool, out_size),
    ])
    return model


def create_network_4conv(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, dropout=0.5):
    if internal_size <= 0:
        internal_size = inp_size
    out_conv = internal_size
    for i in range(4):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / 1 + 1)

    # out_pool = int((out_conv + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1)
    model = Sequential('x, edge_index, edge_weights?', [
        (GCNConv(inp_size, internal_size), 'x, edge_index, edge_weights -> x'),
        Reshape(1, internal_size),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size),
        ReLU(inplace=True),
        Flatten(),
        Dropout(dropout),
        Linear(out_conv, out_size),
    ])
    return model
