import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from mlxtend.plotting import heatmap
import os
import matplotlib.ticker as ticker
from gnn.loaders.util import split_dataset_graph
import pathlib
from gnn.loaders.load import load_data
import pandas as pd
import networkx as nx

import pylab
os.environ["DGLBACKEND"] = "pytorch"  # tell DGL what backend to use
import dgl

def get_loss_map(loss_fn, x, y):
  print(type(x))

  """Maps the loss function on a 100-by-100 grid between (-5, -5) and (8, 8)."""
  losses = [[0.0] * 1001 for _ in range(1001)]
  x = torch.from_numpy(x[0])
  y = torch.from_numpy(np.asarray(y[0]))
  for wi in range(1001):
    for wb in range(1001):
      w = -5.0 + 13.0 * wi / 1000.0
      b = -5.0 + 13.0 * wb / 1000.0
      ywb = x * w + b
      losses[wi][wb] = loss_fn(ywb, y).item()

  return list(reversed(losses))  # Because y will be reversed.

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

    if len(dataset) > 1:
        data = [x for x in dataset]  # .to(device)
    else:
        data = dataset[0]  # .to(device)

    return data, dataset.num_features


class GatedGCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, activation=torch.sigmoid):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        self.activation = activation

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
        # print(e_j)
        # print(type(self.activation))
        # if activation == "sigmoid":
        #     σ_j = self.activation(e_j)
        # else:
        σ_j = self.activation()
        σ_j = σ_j(e_j)
        # print(torch.max(σ_j))
        # print(torch.min(σ_j))
        # print(Ax)

        #

        # print(type(σ_j))

        # h = Ax + Σ_j η_j * Bxj

        h = Ax + torch.nan_to_num(torch.sum(σ_j * Bx_j, dim=1) / torch.sum(σ_j, dim=1))
        # print("h===",h)
        # print(torch.nan_to_num(h))

        # prints
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

    def __init__(self, input_dim, hidden_dim, output_dim, L, activation=torch.sigmoid):
        super().__init__()
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList(
            [
                GatedGCNLayer(hidden_dim, hidden_dim, activation=activation)
                for _ in range(L)
            ]
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
    models_tr = [[model.embedding_h.weight, model.embedding_h.bias]]
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(
        data_loader
    ):
        batch_X = batch_graphs.ndata["feat"]
        batch_E = batch_graphs.edata["feat"]


        batch_scores = model(
            batch_graphs, batch_X, batch_E, batch_snorm_n, batch_snorm_e
        )

        models_tr.append([model.cpu().embedding_h.weight, model.cpu().embedding_h.bias])
        J = loss(batch_scores, batch_labels)
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        epoch_loss += J.detach().item()
        nb_data += batch_labels.size(0)
    # print("epoch loss ==", epoch_loss)
    epoch_loss /= iter + 1

    return epoch_loss, models_tr


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
    num_steps=5,
    select_size=(100, 100),
    sort_labels=True,
    plot=True,
    save_name="save.png",
):
    model.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    models = [[model.embedding_h.weight, model.embedding_h.bias]]


    (
        batch_graphs,
        batch_labels,
        batch_snorm_n,
        batch_snorm_e,
    ) = collate(data_loader)
    # print(batch_graphs.device)
    # print(batch_snorm_n.device)
    # print(batch_snorm_e.device)
    bg = batch_graphs.to('cuda:0')
    bsn = batch_snorm_n.to('cuda:0')
    bse = batch_snorm_e.to('cuda:0')
    bl = batch_labels.to('cuda:0')
    # print(bg.device)
    # print(bsn.device)
    # print(bse.device)
    inp_size = batch_graphs.ndata["feat"].shape

    x = torch.rand(inp_size, requires_grad=True, device=device)

    optim.add_param_group({"params": x})

    for epoch in range(num_steps):
        batch_E = batch_graphs.edata["feat"]
        be =  batch_E.to('cuda:0')
        # print(be)
        batch_X = x
        # print(batch_X.device)
        batch_scores = model.cuda()(
            bg, batch_X.cuda(), be, bsn, bse
        )
        # print(bl.device)
        # print(batch_scores.device)
        models.append([model.cpu().embedding_h.weight, model.cpu().embedding_h.bias])
        model.to(device)
        J = loss_func(batch_scores, bl)
        optim.zero_grad()
        J.backward()
        optim.step()

        print("Reverse epoch: {:03d}, Loss: {:.10f}".format(epoch, J))

    y_labels = batch_labels.detach().squeeze().numpy()
    x_batched = dgl.unbatch(batch_graphs)
    gen_Xvals = []
    for gr_x in x_batched:
        gen_Xvals.append(gr_x.ndata["feat"])

    range_size = int(len(gen_Xvals) // len(y_labels))

    selections = np.array(range(0, len(gen_Xvals), range_size if range_size > 0 else 1))

    X = x.cpu().detach().numpy()[selections, :]
    # values = np.where(X > 0.5, 1, 0)
    values = X
    val_labels = zip(values, y_labels)
    values, y_labels = zip(*sorted(list(val_labels), key=lambda k: k[1]))
    if plot:
        # Manhattan plot with col average
        df = pd.DataFrame(data=values)

        # df["y"] = y_labels

        # corr = df.corr()
        # reg_corr = corr[:-1]

        # df.mean().plot(style=".")
        # plt.show()
        # plt.savefig(save_name)
        ####Manhattan###
        ###Terrain

        # losses = get_loss_map(loss_func, values, y_labels)
        # cm = pylab.get_cmap('terrain')
        #
        # fig, ax = plt.subplots()
        # plt.xlabel('Bias')
        # plt.ylabel('Weight')
        #
        # i = ax.imshow(losses, cmap=cm, interpolation='nearest', extent=[-5, 8, -5, 8])
        # model_weights, model_biases = zip(*models)
        # mw = torch.tensor(model_weights[0], device='cpu')
        # print(mw)
        # mb = torch.tensor(model_biases, device='cpu')
        # ax.scatter(model_biases, model_weights, c='r', marker='+')
        # ax.plot(model_biases, model_weights, c='r')
        #
        # fig.colorbar(i)
        # plt.show()
        # #####Terrain

    return values


def main(filename, epochs, activation=torch.sigmoid):
    train_data, test_data, num_features = create_graphs(filename)

    train_loader = DataLoader(
        train_data, batch_size=50, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_data, batch_size=50, shuffle=False, collate_fn=collate
    )

    model = GatedGCN(
        input_dim=num_features, hidden_dim=100, output_dim=1, L=3, activation=activation
    )
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3.476e-05)

    testloss = []
    trnloss = []
    for epoch in range(epochs):
        train_loss,modelstr = train(model, optimizer, train_loader, loss)
        test_loss = evaluate(model, optimizer, test_loader, loss)
        testloss.append(test_loss)
        trnloss.append(train_loss)
        print(
            f"Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}"
        )

        # ############ploitting#########################
        # model_weights, model_biases = zip(*modelstr)
        # mw = torch.tensor(model_weights[0], device='cpu')
        # print(mw.size)
        # plt.plot(mw)
        # plt.show()
                # print(mw)
        ############ploitting#########################
    # df = pd.DataFrame(testloss)
    # df.to_csv("output/testloss_" + activation.__name__+ ".csv", index = False)
    # print("here")

    plt.plot(trnloss, label='train_loss')
    plt.plot(testloss,label='val_loss')
    plt.title(label="Losses plotted for " + activation.__name__ + " activation")
    plt.legend()
    plt.show()
    x = (
        reverse(
            model,
            torch.optim.Adam(model.parameters(), lr=0.0001),
            torch.Size([num_features, len(train_data)]),
            train_data,
            save_name=f"{filename}-{epochs}-{activation.__name__}.png"
        )
        # .cpu()
        # .detach()
        # .numpy()
    )
    # print(x)


if __name__ == "__main__":
    for activation in [

        # nn.LogSigmoid,
        # nn.Sigmoid,
        # nn.Softplus,
        # nn.Softmin,
        # nn.Softmax,
        # nn.Softmax2d,
        nn.LogSoftmax,
    ]:
        print(activation)
        main("MiceBL.csv", 250, activation=activation)
