import copy
import math
import os
import pathlib
import random
import shutil
import time
from matplotlib import cm
import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, r2_score
from torch.nn import Identity
import seaborn as sns
from torchsummary import summary
# from torchviz import make_dot
# import h5py
from load import load_data
from load_ensembles2 import load_data_ensembles2
import scipy
import gc

gc.collect()
from networks import create_network_conv, create_network_two_no_conv_relu_dropout, Ensemble, create_network_no_conv, \
    create_network_graph_conv_two_dropout, create_network_no_graph, create_network_pre_conv_dropout, \
    create_network_two_no_conv_dropout, create_network_test, create_ensemble_creator, create_network_no_conv_dropout, \
    create_network_no_conv_relu_dropout, create_network_two_no_conv, create_network_two_no_conv_relu, \
    create_network_no_conv_relu, create_network_2D_graph, create_network_conv2D,create_network_conv1D


# import scipy




# from load_merge import load_data as load_data_merge

######################
def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 100, 100, 100
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)
#####################
def model_creator(model_type, *args, **kwargs):
    def creator():
        return model_type(*args, **kwargs)
    return creator


def optimizer_creator(opt_type, *args, **kwargs):
    def creator(model):
        return opt_type(model.parameters(), *args, **kwargs)
    return creator




def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def save_output(name, output, directory="output"):
    """ TODO: load headers to append new ones if needed """
    ext = ".csv" if name[-4:] != ".csv" else ""
    name = f"{directory}/{name}{ext}"
    keys = sorted(output.keys())
    out_df = pd.DataFrame([output])
    if os.path.exists(name):
        existing_df = df = pd.read_csv(name)
        missing_columns = set(keys) - set(existing_df.columns)
        for c in missing_columns:
            existing_df[c] = None
        missing_columns = set(existing_df.columns) - set(keys)
        for c in missing_columns:
            out_df[c] = None
    else:
        existing_df = pd.DataFrame(columns=keys)
    new_df = pd.concat([existing_df, out_df], ignore_index=True)
    new_df.to_csv(name, index=False)


def save_dataset_info(datasets, test_case="default"):
    data_info = []
    for no, dataset in enumerate(datasets):
        if not hasattr(dataset, "edge_index"):
            continue

        edges_raw = dataset.edge_index[0][0] if isinstance(dataset.edge_index, (tuple, list)) else dataset.edge_index
        edges_raw = edges_raw.numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]

        if hasattr(dataset, "test") and dataset.test is not None:
            y = dataset.test.y
        else:
            y = dataset.y
        G = nx.Graph()
        G.add_edges_from(edges)
        degree = [x[1] for x in nx.degree(G)]
        if hasattr(dataset, "edge_weight") and dataset.edge_weight is not None:
            weights = dataset.edge_weight
        else:
            weights = [0]
        average_y = float(sum(y)/len(y))
        min_y = float(min(y))
        max_y = float(max(y))
        data_info.append({
            "dataset": f"{test_case}-{no}",
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "min_degree": min(degree),
            "max_degree": max(degree),
            "mean_degree": sum(degree)/len(degree),
            "average_weight": float(sum(weights)/len(weights)),
            "min_weight": float(min(weights)),
            "max_weight": float(max(weights)),
            "average_y": average_y,
            "min_y": min_y,
            "max_y": max_y,
            "rmse_average": (sum((_y-average_y)**2 for _y in y)/len(y)),
            "rmse_line": (sum((_y-(min_y+i*(max_y-min_y)/len(y)))**2 for i, _y in enumerate(sorted(y)))/len(y))
        })

    if hasattr(datasets, "valid") and hasattr(datasets.valid[0], "edge_index"):
        edges_raw = datasets.valid[0].edge_index[0][0] if isinstance(datasets.valid[0].edge_index,
                                                    (tuple, list)) else datasets.valid[0].edge_index
        edges_raw = edges_raw.numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        G = nx.Graph()
        G.add_edges_from(edges)
        degree = [x[1] for x in nx.degree(G)]
        datasets.valid[0]

        if hasattr(datasets.valid[0], "edge_weight") and datasets.valid[0].edge_weight is not None:
            weights = datasets.valid[0].edge_weight
        else:
            weights = [0]
        y = datasets.valid[0].y
        average_y = float(sum(y)/len(y))
        min_y = float(min(y))
        max_y = float(max(y))
        data_info.append({
            "dataset": f"{test_case}-validation",
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "min_degree": min(degree),
            "max_degree": max(degree),
            "mean_degree": sum(degree)/len(degree),
            "average_weight": float(sum(weights)/len(weights)),
            "min_weight": float(min(weights)),
            "max_weight": float(max(weights)),
            "average_y": average_y,
            "min_y": min_y,
            "max_y": max_y,
            "rmse_average": (sum((_y-average_y)**2 for _y in y)/len(y)),
            "rmse_line": (sum((_y-(min_y+i*(max_y-min_y)/len(y)))**2 for i, _y in enumerate(sorted(y)))/len(y))
        })

    pd.DataFrame(data_info).to_csv(f"output/data-info-{time.time()}-{test_case}.csv", index=False)


def plot_dataset(datasets, test_case="default"):

    # if not isinstance(datasets, list):
    #     datasets = [datasets]

    # datasets = datasets.copy()
    for no, dataset in enumerate(datasets):
        if not hasattr(dataset, "edge_index"):
            continue
        edges_raw = dataset.edge_index[0][0] if isinstance(dataset.edge_index, (tuple, list)) else dataset.edge_index
        edges_raw = edges_raw.numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        # edges = [(x, y) for x, y in edges if x < 250 and y < 250]
        G = nx.Graph()
        # G.add_nodes_from(list(range(np.max(edges_raw))))
        G.add_edges_from(edges)
        plt.rcParams["figure.figsize"] = (55, 50)
        # matplotlib.pyplot.imshow(A.todense())

        # partition = community.best_partition(G)
        # communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)

        # pos = nx.spring_layout(G, k=0.8)
        # plt.rcParams["figure.figsize"] = (55, 50)

        #plt.subplot(111)

        # labels = [n.numpy()[0] for n in dataset.data.y][:len(G)]
        # set_node_community(G, communities)
        # set_edge_community(G)
        # external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
        # internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
        # internal_color = [(0.5,0.5,0.5) for e in internal]
        # node_color = list(partition.values())#[get_color(G.nodes[v]['community']) for v in G.nodes]
        # pos = community_layout(G, partition)
        options = {
            # 'node_size': 30,
            # 'width': 0.2,
            # 'with_labels': False
        }
        # nx.draw_networkx(
        #     G,
        #     pos=pos,
        #     node_size=0,
        #     edgelist=external,
        #     edge_color="silver",
        #     node_color=node_color,
        #     alpha=0.02,
        #     with_labels=False)
        # # internal edges
        # nx.draw_networkx(
        #     G, pos=pos,
        #     edgelist=internal,
        #     edge_color=internal_color,
        #     node_color=node_color,
        #     alpha=0.8,
        #     with_labels=False)
       # nx.draw_networkx_nodes(G, pos, node_color=list(partition.values()), cmap=plt.cm.tab10, **options)
        #nx.draw_networkx_edges(G, pos, alpha=0.2)
        # nx.draw(G,  node_color=labels, cmap=plt.cm.tab10, font_weight='bold', **options)

        plt.savefig(f"images/{test_case}-{no}-neighbourhood_graph.png")
        plt.close()
        #plot embeddings-

        # reduced_dim = TSNE(3, metric='cosine').fit_transform(dataset.edge_weight.reshape(-1, 1))
        # data_scale = pd.DataFrame(reduced_dim, columns=['tsne1', 'tsne2','tsne3',])
        # plt.figure(figsize=(10, 8))
        # plt.plot(reduced_dim[:, 0], reduced_dim[:, 1], 'r.')
        # plt.xlabel('TSNE 1');
        # plt.ylabel('TSNE 2');
        # plt.title("Embeddings Visualized with TSNE");
        # plt.show()
        #
        # kmeans_tsne_scale = KMeans(n_clusters=10, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(
        #     data_scale)
        # # print('KMeans tSNE Scaled Silhouette Score: {}'.format(
        # #     silhouette_score(data_scale, kmeans_tsne_scale.labels_, metric='cosine')))
        # # data_scale['clusters'] = pd.Series(predict, index=df.index)
        # # genre_counts = count_items(genres)
        # labels_tsne_scale = kmeans_tsne_scale.labels_
        # clusters_tsne_scale = pd.concat([data_scale, pd.DataFrame({'tsne_clusters': labels_tsne_scale})], axis=1)
        #
        # print( clusters_tsne_scale.groupby(["tsne_clusters"]).count())
        #
        # plt.figure(figsize=(15, 15))
        # sns.scatterplot(x=clusters_tsne_scale.iloc[:, 0], y=clusters_tsne_scale.iloc[:, 1], hue=labels_tsne_scale,
        #                 palette='Set1', alpha=0.6).set_title('Cluster Vis tSNE Scaled Data', fontsize=15)
        # plt.legend()
        # plt.show()
        #print(dataset.edge_weight.cpu().detach().numpy())

        # G_weight = nx.from_numpy_array(dataset.edge_weight.cpu().detach().numpy().reshape(-1, 1))
        # partition_W = community.best_partition(G_weight)
        # pos = community_layout(G_weight, partition+W)
        # nx.draw_networkx_nodes(G_weight, pos, node_color=list(partition_W.values()), cmap=plt.cm.tab10, **options)
        # nx.draw_networkx_edges(G_weight, pos, alpha=0.2)
        # plt.show()
        # prints

    if hasattr(datasets, "valid") and hasattr(datasets.valid[0], "edge_index"):
        edges_raw = datasets.valid[0].edge_index[0][0] if isinstance(datasets.valid[0].edge_index,
                                                                (tuple, list)) else datasets.valid[0].edge_index
        edges_raw = edges_raw.numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        # edges = [(x, y) for x, y in edges if x < 250 and y < 250]
        G = nx.Graph()
        # G.add_nodes_from(list(range(np.max(edges_raw))))
        G.add_edges_from(edges)
        plt.rcParams["figure.figsize"] = (55, 50)

        partition = community.best_partition(G)
        pos = community_layout(G, partition)
        options = {
        }

        nx.draw_networkx_nodes(G, pos, node_color=list(partition.values()), cmap=plt.cm.tab10, **options)
        nx.draw_networkx_edges(G, pos, alpha=0.2)

        plt.savefig(f"images/{test_case}-validation-neighbourhood_graph.png")
        plt.close()



def l1_regularize(model, loss, l1_lambda):
    if hasattr(model, "l1_regularize"):
        return model.l1_regularize(loss, l1_lambda)
    l1_parameters = []
    for parameter in model.parameters():
        l1_parameters.append(parameter.view(-1))

    l1 = l1_lambda * torch.abs(torch.cat(l1_parameters)).sum().float()

    return loss + l1


def validate(model, data, loss_func=torch.nn.MSELoss()):
    pred = model(data["x"], torch.tensor([[1], [1]]))
    loss = loss_func(pred, data["y"])
    return pred, float(loss)


def train_none(*args, **kwargs):
    print(f"Would have trained with {args}, {kwargs}")
    return [0], 0


def train_learn(m, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, **_):
    loss_func = torch.nn.MSELoss()
    seed_size = 10
    n_best = 5
    total = 20
    noise_level = 10
    search_noise = 0.01
    mutation_rate = 0.05

    losses = []
    train_losses = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"
    models = [copy.deepcopy(m) for _ in range(seed_size)]
    params = [{"lr": optimizer.defaults["lr"]+(random.random()-0.5)/noise_level,
               "wd": optimizer.defaults["weight_decay"]+(random.random()-0.5)/noise_level,
               "lambda": l1_lambda+(random.random()-0.5)/noise_level} for _ in range(seed_size)]
    result = []
    last_result = []
    for epoch in range(epochs):
        if len(result) < len(last_result):
            result.extend([max(last_result+result) if len(last_result) else None]*(len(last_result)-len(result)))
        if len(last_result) > 0:
            best_ones = sorted(list(enumerate([abs(x)-abs(y) for x, y in zip(result, last_result)])), key=lambda x: x[1], reverse=True)[:n_best]
            best_models = sorted(list(enumerate([abs(x) for x in last_result])), key=lambda x: x[1])[:n_best]
            best_ones = best_ones
            new_models = []
            new_models.extend([copy.deepcopy(models[x]) for x, _ in best_ones])
            new_params = []
            new_params.extend([params[x] for x, _ in best_ones])

            # new_models.extend([copy.deepcopy(models[x]) for x, _ in best_models])
            # new_params.extend([params[x] for x, _ in best_ones])

            choices = random.choices(population=[x[0] for x in best_ones], weights=[x[1] for x in best_ones], k=(total-len(new_models)))

            new_models.extend([copy.deepcopy(models[i]) for i in choices])
            new_params.extend([
                {key: value + random.random()*search_noise if random.random() < mutation_rate/len(params[i]) else value for key, value in params[i].items()} for i in choices
            ])

            models = new_models
            params = new_params
            result = last_result
            last_result = []
        test_losses = []
        for m_train, param in zip(models, params):
            optimizer = torch.optim.Adam(m_train.parameters(), lr=abs(param.get("lr", 0.01)), weight_decay=abs(param.get("wd", 0.01)))

            l1_lambda = param.get("lambda", 0.01)

            m_train.train()
            optimizer.zero_grad()
            out = m_train((data.x, data.edge_index))

            m_train.eval()
            if not hasattr(data, "range"):
                y = data.y
                if len(data.test.x) > 0:
                    out_test = m_train((data.test.x, data.test.edge_index))
                    test_y = data.test.y
                else:
                    out_test = m_train((data.x, data.edge_index))
                    test_y = data.y
            else:
                out_test = out[data.test.range.start: data.test.range.stop]
                out = out[data.range.start: data.range.stop]
                y = data.y[data.range.start: data.range.stop]
                test_y = data.y[data.test.range.start: data.test.range.stop]

            loss = loss_func(out, y)

            loss.backward()
            optimizer.step()

            test_losses.append(float(loss_func(out_test, test_y)))
            last_result.append(float(loss))

        min_index = test_losses.index(min(test_losses))
        train_losses.append(last_result[min_index])
        losses.append(test_losses[min_index])
        if test_loss < least_loss:
            no_improvement = 0
            least_loss = test_loss
        else:
            no_improvement += 1

        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}, Params: {}'.format(epoch, train_losses[-1], losses[-1], params[min_index]))
        if 0 < max_no_improvement <= no_improvement:
            break

    pd.DataFrame({"train_loss": train_losses, "loss": losses}).to_csv(f"output/{test_name}-train-learn-data.csv")
    return losses, epoch


def train_batch(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, save_loss=True, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"

    train_data, test_data = data
    for epoch in range(epochs):
        train_loss_batch = []
        test_loss_batch = []
        for i_batch, sample_batched in enumerate(train_data):
            model.train()
            optimizer.zero_grad()
            out = model(sample_batched["input"])

            model.eval()
            y = sample_batched["output"]

            out_test = model(test_data[:]["input"])
            test_y = test_data[:]["output"]

            # if plot and epoch == 1:
            #     size = batch_size if batch_size > 0 else 100
            #     plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
            #     plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
            #     plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
            #     plt.xlabel("Subject")
            #     plt.ylabel("Value")
            #     plt.legend(loc='upper right')
            #
            loss = loss_func(out, y)#     plt.savefig(f"images/{test_name}-batch-pred-vs-correct-epoch-0.png")
            #     plt.close()

            # corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
            # corrs.append(corr[0][1])
            loss = loss_func(out, y)
            if use_l1_reg:
                loss = l1_regularize(model, loss, l1_lambda)

            loss.backward()
            optimizer.step()
            train_loss_batch.append(float(loss))
            test_loss = (float(loss_func(out_test, test_y)))
            test_loss_batch.append(test_loss)
        losses.append(sum(test_loss_batch) / len(test_loss_batch))
        train_losses.append(sum(train_loss_batch) / len(train_loss_batch))
        corrs.append(0)
        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    # if plot:
    #     size = batch_size if batch_size > 0 else 100
    #     plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
    #     plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
    #     plt.title(f"Predicted and correct (first {size} subjects) - Final")
    #     plt.xlabel("Subject")
    #     plt.ylabel("Value")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
    #     plt.close()
    #
    #     plt.plot(losses, label="Loss")
    #     plt.title("Loss per epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-loss.png")
    #     plt.close()
    #
    #     plt.plot(corrs, label="Correlation")
    #     plt.title("Correlation between prediction and correct per epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Correlation")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-correlation.png")
    #     plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
            f"output/{test_name}-train-data.csv")
    return losses, epoch


def train_batch(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"

    train_data, test_data = data
    for epoch in range(epochs):
        train_loss_batch = []
        test_loss_batch = []
        for i_batch, sample_batched in enumerate(train_data):
            model.train()
            optimizer.zero_grad()
            out = model(sample_batched["input"])

            model.eval()
            y = sample_batched["output"]

            out_test = model(test_data[:]["input"])
            test_y = test_data[:]["output"]

            # if plot and epoch == 1:
            #     size = batch_size if batch_size > 0 else 100
            #     plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
            #     plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
            #     plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
            #     plt.xlabel("Subject")
            #     plt.ylabel("Value")
            #     plt.legend(loc='upper right')
            #
            #     plt.savefig(f"images/{test_name}-batch-pred-vs-correct-epoch-0.png")
            #     plt.close()

            # corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
            # corrs.append(corr[0][1])
            loss = loss_func(out, y)
            if use_l1_reg:
                loss = l1_regularize(model, loss, l1_lambda)

            loss.backward()
            optimizer.step()
            train_loss_batch.append(float(loss))
            test_loss = (float(loss_func(out_test, test_y)))
            test_loss_batch.append(test_loss)
        losses.append(sum(test_loss_batch) / len(test_loss_batch))
        train_losses.append(sum(train_loss_batch) / len(train_loss_batch))
        corrs.append(0)
        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    # if plot:
    #     size = batch_size if batch_size > 0 else 100
    #     plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
    #     plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
    #     plt.title(f"Predicted and correct (first {size} subjects) - Final")
    #     plt.xlabel("Subject")
    #     plt.ylabel("Value")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
    #     plt.close()
    #
    #     plt.plot(losses, label="Loss")
    #     plt.title("Loss per epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-loss.png")
    #     plt.close()
    #
    #     plt.plot(corrs, label="Correlation")
    #     plt.title("Correlation between prediction and correct per epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Correlation")
    #     plt.legend(loc='upper right')
    #     plt.savefig(f"images/{test_name}-gnn-correlation.png")
    #     plt.close()
    pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
        f"output/{test_name}-train-data.csv")
    return losses, epoch


def train_ensemble2(model, optimizer, data, test_case="default", plot=False, epochs=1000, aggregate_epochs=None, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, save_loss=True, early_stopper=lambda _, __, ___: None, print_epochs=True, **_):
    test_name = f"{time.time()}-{test_case}"
    if aggregate_epochs is None:
        aggregate_epochs = epochs
    orig_model = model
    orig_optimizer = optimizer
    paths = []
    losses_dict = {}
    path_aggr = '/home/rags/gnn/gnn/model/aggr/'
    pathlib.Path(f"model/ensemble/{test_name}/").mkdir(parents=True, exist_ok=True)
    min_loss_so_far = 1000000000
    best_model_so_far = None
    for i_batch, sample_batched in enumerate(data[:-1]):
        model = orig_model()
        optimizer = orig_optimizer(model)
        loss_func = torch.nn.MSELoss()
        no_improvement = 0
        epoch = 0
        losses = []
        train_losses = []
        corrs = []
        test_loss = 100000000
        least_loss = test_loss
        best_corr = 0.2
        path = f"model/ensemble/{test_name}/{i_batch}.pt"
        paths.append(path)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            if hasattr(sample_batched, "edge_weight") and sample_batched.edge_weight is not None:  # ** to prevent accessing float on None
                args_tuple = sample_batched.x.float(), sample_batched.edge_index, sample_batched.edge_weight.float()
            else:
                args_tuple = sample_batched.x, sample_batched.edge_index

            out = model(*args_tuple)
            # out = model(sample_batched.x, sample_batched.edge_index)

            model.eval()
            y = sample_batched.y

            if hasattr(sample_batched, "edge_weight") and sample_batched.edge_weight is not None:
                test_tuple = sample_batched.test.x.float(), sample_batched.test.edge_index, sample_batched.test.edge_weight.float()
            else:
                test_tuple = sample_batched.test.x, sample_batched.test.edge_index

            out_test = model(*test_tuple)
            # out_test = model(sample_batched.test.x, sample_batched.test.edge_index)
            test_y = sample_batched.test.y
            pred_val = out_test.detach().squeeze().numpy()
            true_val = test_y.detach().squeeze().numpy()
            df = pd.DataFrame(data=np.column_stack((true_val,pred_val)),columns=['true_val','pred_val'])
            df["true_val_r"] = df.true_val.rank()
            df["pred_val_r"] = df.pred_val.rank()

            corr = np.corrcoef([n.detach().squeeze().numpy() for n in out_test], [n.detach().squeeze().numpy() for n in test_y])
            #corr = np.corrcoef(df.pred_val_r, df.true_val_r)

            corr = corr - np.min(corr) / (np.max(corr) - np.min(corr))

            corrs.append(corr[0][1])
            # r2 = r2_score(out_test.detach().squeeze().numpy(), test_y.detach().squeeze().numpy())
            # n = 40
            # k = 2
            # adj_r2_score = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
            # print(adj_r2_score)
            # total_params = sum(p.numel() for p in model.parameters())
            # print('total params --', total_params)

            if plot and epoch == 1 and False:
                size = batch_size if batch_size > 0 else min((len(out_test), len(test_y), 100))
                plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
                plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
                plt.title(f"Predicted and correct (first {size} subjects) - Batch {i_batch} - Epoch 0")
                plt.xlabel("Subject")
                plt.ylabel("Value")
                plt.legend(loc='upper right')

                plt.savefig(f"images/{test_name}-batch-pred-vs-correct-{i_batch}-epoch-0.png")
                plt.close()


            loss = loss_func(out, y)
            if use_l1_reg:
                loss = l1_regularize(model, loss, l1_lambda)

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))
            test_loss = (float(loss_func(out_test, test_y)))
            losses.append(test_loss)
            if test_loss < least_loss: # it saves the best model udl this is per dataset
                no_improvement = 0
                least_loss = losses[-1]
                # save model
                torch.save(model.state_dict(), path)
            else:
                no_improvement += 1

            if test_loss < min_loss_so_far:
                min_loss_so_far = test_loss
                best_model_so_far = model


            if 0 < max_no_improvement <= no_improvement:
                break
            if print_epochs:
                print('Batch: {:02d}, Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(i_batch, epoch, train_losses[-1], losses[-1]))

        def cleanup():
            shutil.rmtree(f"model/ensemble/{test_name}", ignore_errors=True)
        early_stopper(min(losses), i_batch, cleanup)

        losses_dict[i_batch] = losses
        if plot and False:
            size = batch_size if batch_size > 0 else min((len(out_test), len(test_y), 100))
            plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
            plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
            plt.title(f"Predicted and correct (first {size} subjects) - Batch {i_batch} - Final")
            plt.xlabel("Subject")
            plt.ylabel("Value")
            plt.legend(loc='upper right')
            plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-batch-{i_batch}-epoch-final.png")
            plt.close()

            plt.plot(losses, label="Loss")
            plt.title("Loss per epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.savefig(f"images/{test_name}-gnn-batch-{i_batch}-loss.png")
            plt.close()

            plt.plot(losses, label="Loss")
            plt.plot(corrs, label="Correlation")
            plt.title("Correlation and loss between prediction and correct per epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Correlation")
            plt.legend(loc='upper right')
            plt.savefig(f"images/{test_name}-gnn-batch-{i_batch}-correlation.png")
            plt.close()
        if save_loss:
            pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
                f"output/{test_name}-train-ensemble2-{i_batch}-data.csv")
    final_outs = []
    loss_func = torch.nn.MSELoss()

    losses_and_epochs = {}
    for k, k_losses in losses_dict.items():
        losses_and_epochs[f"min_loss_{k}"] = min(k_losses)
        losses_and_epochs[f"min_epoch_{k}"] = k_losses.index(losses_and_epochs[f"min_loss_{k}"])

    if hasattr(data[-1], "edge_weight") and data[-1].edge_weight is not None:
            valid_tuple = data[-1].x.float(), data[-1].edge_index, data[-1].edge_weight.float()
    else:
        valid_tuple = data[-1].x, data[-1].edge_index

    models = []
    size = 1

    def get_size(_, __, output):
        nonlocal size
        size = output.shape[1]

    for path in paths:
        model = orig_model()
        model.load_state_dict(torch.load(path))
        models.append(model)
        handle = model.get(-2).register_forward_hook(get_size)
        model.eval()
        final_outs.append([float(x) for x in model(*valid_tuple)])
        handle.remove()
        model.replace(-1, Identity())
    aggr_model = Ensemble(models, size)
    optimizer = orig_optimizer(aggr_model)
    aggr_losses = []
    aggr_train_losses = []
    corrs = []
    train_data = data[-2]
    aggr_path = f"model/ensemble/{test_name}/aggr.pt"
    least_loss = 1000000000000
    for aggr_epoch in range(aggregate_epochs):
        aggr_model.train()
        optimizer.zero_grad()

        if hasattr(train_data, "edge_weight") and train_data.edge_weight is not None:
            valid_tuple = train_data.x.float(), train_data.edge_index, train_data.edge_weight.float()
        else:
            valid_tuple = train_data.x, train_data.edge_index

        out = aggr_model(*valid_tuple)

        aggr_model.eval()
        y = train_data.y

        if hasattr(train_data.test, "edge_weight") and train_data.edge_weight is not None:
            valid_tuple = train_data.test.x.float(), train_data.test.edge_index, train_data.test.edge_weight.float()
        else:
            valid_tuple = train_data.test.x, train_data.test.edge_index

        out_test: torch.Tensor = aggr_model(*valid_tuple)
        test_y = train_data.test.y

        loss = loss_func(out, y)
        aggr_train_losses.append(float(loss))
        if use_l1_reg:
            loss = l1_regularize(aggr_model, loss, l1_lambda)

        df_out = pd.DataFrame({"predict": [x[0] for x in out_test.tolist()], "actual": [x[0] for x in test_y.tolist()]})
        df_out.to_csv(f"output/predictions-{test_name}.csv", index=False)
        test_loss = (float(loss_func(out_test, test_y)))

        if test_loss < least_loss:  # it saves the best model udl this is per dataset
            least_loss = test_loss
            torch.save(aggr_model.state_dict(), aggr_path)

        loss.backward()
        optimizer.step()
        aggr_losses.append(float(test_loss))

        corr = np.corrcoef([n.detach().squeeze().numpy() for n in out_test],
                           [n.detach().squeeze().numpy() for n in test_y])
        corrs.append(corr[0][1])

        if print_epochs:
            print('Batch: aggregation, Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(aggr_epoch, loss,
                                                                                                test_loss))

    # save aggr_model
    # print('saving')
    aggr_model.load_state_dict(torch.load(aggr_path))
    torch.save(aggr_model, aggr_path)
    if plot:
        # plt.plot(aggr_losses, label="Loss")
        # plt.plot(corrs, label="Correlation")
        # plt.title("Aggregation Correlation (Spearman) and loss between prediction and correct per epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Correlation")
        # plt.legend(loc='upper right')
        # plt.savefig(f"images/{test_name}-gnn-batch-aggr-correlation.png")
        # plt.close()

        fig, ax = plt.subplots()
        # make a plot
        ls = ax.plot(aggr_losses,color="red", label="Loss")
        # set x-axis label
        ax.set_xlabel("Epochs", fontsize=14)
        # set y-axis label
        ax.set_ylabel("MSE Loss", color="red", fontsize=14)
        #ax.legend(loc="upper left")
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        cr = ax2.plot(corrs, color="blue",label="Correlation")
        ax2.set_ylabel("Correlation", color="blue", fontsize=14)
        #ax2.legend(loc="upper right")
        lns = ls + cr
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 0.5), ncol=2, borderaxespad=0.25)

        # save the plot as a file
        fig.savefig(f"images/{test_name}-gnn-batch-aggr-correlation.png",bbox_inches='tight')
        plt.close()
    if hasattr(data[-1], "edge_weight") and data[-1].edge_weight is not None:
        valid_tuple = data[-1].x.float(), data[-1].edge_index, data[-1].edge_weight.float()
    else:
        valid_tuple = data[-1].x, data[-1].edge_index

    pred_val = best_model_so_far(*valid_tuple)
    aggr_val = aggr_model(*valid_tuple)
    best_loss = float(loss_func(pred_val, data[-1].y.flatten()))
    aggr_loss = float(loss_func(aggr_val, data[-1].y.flatten()))
    avg_val = [sum(x)/len(x) for x in zip(*final_outs)]
    loss = float(loss_func(torch.tensor(avg_val), data[-1].y.flatten()))
    avg_corr = np.corrcoef(avg_val, [n.detach().numpy() for n in data[-1].y.flatten()])
    avg_corr = avg_corr[0][1]
    corr = np.corrcoef([n.detach().squeeze().numpy() for n in pred_val], [n.detach().numpy() for n in data[-1].y.flatten()])
    best_corr = (corr[0][1])

    #her = np.cov(pred_val.detach().numpy(),rowvar=0)/(np.cov(pred_val.detach().numpy(),rowvar=0) + np.cov(data[-1].y.flatten()))
    VarPhen = np.cov(data[-1].y.flatten())
    err = pred_val - data[-1].y.flatten()
    VarErr = np.cov(err.detach().numpy())
    VarGen = VarPhen - VarErr
    her = VarGen / VarPhen

    shutil.rmtree(f"model/ensemble/{test_name}", ignore_errors=True)
    return {
            "basics": {
                "losses": list(sorted(losses_dict.values(), key=lambda x: min(x))[-1]),
                "epoch": epochs,
                "all_losses": sum(losses_dict.values(), [loss]),
                "best_loss": best_loss,
                "aggr_loss": aggr_loss,
                "min_loss_validation": loss,
                "model": best_model_so_far,
                "aggr_model": aggr_model,
                "cors": corrs,
                "cor_best": best_corr,
                "cor_valid": avg_corr,
                "heritabilty" : np.mean(her),
                **losses_and_epochs
            }
    }


def train_no_edge(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, save_loss=True, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        model.eval()
        if not hasattr(data, "range"):
            y = data.y
            if len(data.test.x) > 0:
                out_test = model(data.test.x)
                test_y = data.test.y

            else:
                out_test = model(data.x)
                test_y = data.y

        else:
            out_test = out[data.test.range.start: data.test.range.stop]
            out = out[data.range.start: data.range.stop]
            y = data.y[data.range.start: data.range.stop]
            test_y = data.y[data.test.range.start: data.test.range.stop]

        if plot and epoch == 1:
            size = batch_size if batch_size > 0 else 100
            plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
            plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
            plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
            plt.xlabel("Subject")
            plt.ylabel("Value")
            plt.legend(loc='upper right')

            plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-0.png")
            plt.close()

        #corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
        #corr = np.corrcoef([n.cpu().detach() for n in out_test], [n.cpu().detach() for n in test_y])
        corr = np.corrcoef([n.detach().squeeze().numpy() for n in out_test],
                           [n.detach().squeeze().numpy() for n in test_y])
        corrs.append(corr[0][1])

        loss = loss_func(out, y)
        if use_l1_reg:
            loss = l1_regularize(model, loss, l1_lambda)

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        test_loss = (float(loss_func(out_test, test_y)))

        losses.append(test_loss)
        train_losses.append(float(loss))
        corrs.append(0)
        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    if plot:
        size = batch_size if batch_size > 0 else 100
        plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
        plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
        plt.title(f"Predicted and correct (first {size} subjects) - Final")
        plt.xlabel("Subject")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
        plt.close()

        plt.plot(losses, label="Loss")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-loss.png")
        plt.close()

        plt.plot(corrs, label="Correlation")
        plt.title("Correlation between prediction aTruend correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-correlation.png")
        plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
            f"output/{test_name}-train-data.csv")

    return {
                "basics": {
                    "losses": losses,
                    "epoch": epochs,
                    "cors": corrs,
                }
    }
#
# def batch(data, size=0):
#     if size == 0:
#         return [data]
#
#     def create_range(data, idx):
#         n_data = type("Data", (object,), {})
#         n_data.test = type("Test", (object,), {})
#         n_data.x = data.x.index_select(0, idx)
#         n_data.y = data.y.index_select(0, idx)
#         n_data.edge_index = data.edge_index
#
#         # n_data.range = range(start, stop)
#         # n_data.test.range = range(train_len, train_len+test_len)
#         n_data.test.x = data.test.x
#         n_data.test.y = data.test.y
#         n_data.test.edge_index = data.test.edge_index
#         return n_data
#
#     def iterator(data, start, stop, size):
#         idxs = list(range(start, stop))
#         random.shuffle(idxs)
#         idxs = torch.tensor(idxs, dtype=torch.int32)
#         for start in range(start, stop, size):
#             yield create_range(data, idxs[start:start+size])
#
#     if hasattr(data, "range"):
#         return [create_range(data, start, start+size, data.range.stop, data.test.range.stop) for start in range(0, data.range.stop, size)]
#     else:
#         return iterator(data, 0, len(data.x), size)
#         n_data = type("Data", (object,), {})
#         n_data.test = type("Test", (object,), {})
#         n_data.test.x = data.test.x
#         n_data.test.y = data.test.y
#         n_data.test.edge_index = data.test.edge_index
#         n_data.x = torch.cat([data.x, data.test.x])
#         n_data.y = torch.cat([data.y, data.test.y])
#         n_data.edge_index = torch.cat([data.edge_index, data.test.edge_index], dim=1)
#         return [create_range(n_data, start, start+size, len(data.x), len(data.test.x)) for start in range(0, len(data.x), size)]

def train(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, save_loss=True, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data.to(device)
    for epoch in range(epochs):
        model.to(device).train()
        #model.train()

        optimizer.zero_grad()
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            arg_tuple = data.x.float(), data.edge_index, data.edge_weight.float()
        else:
            arg_tuple = data.x, data.edge_index

        out = model(arg_tuple).to(device)
        # tsne = TSNE(2, verbose=1)
        # tsne_proj = tsne.fit_transform(out)
        # cmap = cm.get_cmap('tab20')
        # fig, ax = plt.subplots(figsize=(8, 8))
        # num_categories = 11
        # for lab in range(num_categories):
        #     indices = test_targets == lab
        #     ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
        #                alpha=0.5)
        # ax.legend(fontsize='large', markerscale=2)
        # plt.show()
        # print(model.eval())
        model.eval()

        if not hasattr(data, "range"):
            y = data.y
            if len(data.test.x) > 0:
                if hasattr(data.test, "edge_weight") and data.test.edge_weight is not None:
                    test_tuple = data.test.x.float(), data.test.edge_index, data.test.edge_weight.float()
                else:
                    test_tuple = data.test.x, data.test.edge_index
                out_test = model(test_tuple)
                test_y = data.test.y
            else:
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    test_tuple = data.x, data.edge_index, data.edge_weight
                else:
                    test_tuple = data.x, data.edge_index
                out_test = model(test_tuple)
                test_y = data.y
        else:
            out_test = out[data.test.range.start: data.test.range.stop]
            out = out[data.range.start: data.range.stop]
            y = data.y[data.range.start: data.range.stop]
            test_y = data.y[data.test.range.start: data.test.range.stop]

        if plot and epoch == 1:
            size = batch_size if batch_size > 0 else 100
            plt.scatter(range(size), [n.detach().cpu() for n in out_test[:size]], label="Predicted")
            plt.scatter(range(size), [n.detach().cpu() for n in test_y][:size], label="Correct")
            plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
            plt.xlabel("Subject")
            plt.ylabel("Value")
            plt.legend(loc='upper right')

            plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-0.png")
            plt.close()

        corr = np.corrcoef([n.detach().squeeze().cpu().numpy() for n in out_test],
                           [n.detach().squeeze().cpu().numpy() for n in test_y])

        #corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
        corrs.append(corr[0][1])

        loss = loss_func(out, y)
        if use_l1_reg:
            loss = model.l1_regularize(loss, l1_lambda)
        loss.backward()
        optimizer.step()

        test_loss = (float(loss_func(out_test, test_y)))

        losses.append(test_loss)
        train_losses.append(float(loss))
        # corrs.append(0)
        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    if plot:
        size = batch_size if batch_size > 0 else 100
        plt.scatter(range(size), [n.detach().cpu() for n in out_test][:size], label="Predicted")
        plt.scatter(range(size), [n.detach().cpu() for n in test_y][:size], label="Correct")
        plt.title(f"Predicted and correct (first {size} subjects) - Final")
        plt.xlabel("Subject")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
        plt.close()

        plt.plot(losses, label="Loss")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-loss.png")
        plt.close()

        plt.plot(corrs, label="Correlation")
        plt.title("Correlation between prediction and correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-correlation.png")
        plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(f"output/{test_name}-train-data.csv")
    return {
            "basics": {
                "losses": losses,
                "epoch": epochs,
                "cors": corrs,
            }
    }


def train_batches(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, save_loss=True, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"
    for epoch in range(epochs):
        losses.append(0)
        train_losses.append(0)
        for batch_no, datum in enumerate(data):
            model.train()
            optimizer.zero_grad()
            out = model((datum.x, datum.edge_index))

            model.eval()
            if not hasattr(datum, "range"):
                y = datum.y
                if len(datum.test.x) > 0:
                    out_test = model((datum.test.x, datum.test.edge_index))
                    test_y = datum.test.y
                else:
                    out_test = model((datum.x, datum.edge_index))
                    test_y = datum.y
            else:
                out_test = out[datum.test.range.start: datum.test.range.stop]
                out = out[datum.range.start: datum.range.stop]
                y = datum.y[datum.range.start: datum.range.stop]
                test_y = datum.y[datum.test.range.start: datum.test.range.stop]

            if plot and epoch == 1 and batch_no == 0:
                size = 100
                plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
                plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
                plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
                plt.xlabel("Subject")
                plt.ylabel("Value")
                plt.legend(loc='upper right')

                plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-0.png")
                plt.close()

            # corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
            # corrs.append(corr[0][1])

            loss = loss_func(out, y)
            if use_l1_reg:
                loss = model.l1_regularize(loss, l1_lambda)
            loss.backward()
            optimizer.step()

            test_loss = (float(loss_func(out_test, test_y)))

            losses[-1] += test_loss/len(data)
            train_losses[-1] += float(loss)/len(data)

            print('Epoch: {:03d}:{:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, batch_no, loss, test_loss))
        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    if plot:

        size = 100
        plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
        plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
        plt.title(f"Predicted and correct (first {size} subjects) - Final")
        plt.xlabel("Subject")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
        plt.close()

        plt.plot(losses, label="Loss")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-loss.png")
        plt.close()

        plt.plot(corrs, label="Correlation")
        plt.title("Correlation between prediction and correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-correlation.png")
        plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses}).to_csv(f"output/{test_name}-train-data.csv")
    return losses, epoch


def train_ensamble(model, optimizer, data, test_case="default", plot=True, epochs=1000, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, save_loss=True, **_):
    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    epoch = 0
    test_name = f"{time.time()}-{test_case}"
    batches = random.sample(list(range(len(data))), len(data))
    test = data[batches.pop(-1)]
    for epoch in range(epochs):
        losses.append(0)
        train_losses.append(0)
        random.shuffle(batches)
        # batches = random.sample(list(range(len(data)-1)), len(data)-1)
        train_b = batches
        for batch_no in train_b:
            datum = data[batch_no]
            datum.test = test
            model.train()
            optimizer.zero_grad()
            out = model((datum.x, datum.edge_index))

            model.eval()
            if not hasattr(datum, "range"):
                y = datum.y
                if len(datum.test.x) > 0:
                    out_test = model((datum.test.x, datum.test.edge_index))
                    test_y = datum.test.y
                else:
                    out_test = model((datum.x, datum.edge_index))
                    test_y = datum.y
            else:
                out_test = out[datum.test.range.start: datum.test.range.stop]
                out = out[datum.range.start: datum.range.stop]
                y = datum.y[datum.range.start: datum.range.stop]
                test_y = datum.y[datum.test.range.start: datum.test.range.stop]

            if plot and epoch == 1 and batch_no == 0:
                size = 100
                plt.scatter(range(size), [n.detach() for n in out_test[:size]], label="Predicted")
                plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
                plt.title(f"Predicted and correct (first {size} subjects) - Epoch 0")
                plt.xlabel("Subject")
                plt.ylabel("Value")
                plt.legend(loc='upper right')

                plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-0.png")
                plt.close()

            # corr = np.corrcoef([n.detach() for n in out_test], [n.detach() for n in test_y])
            # corrs.append(corr[0][1])

            loss = loss_func(out, y)
            if use_l1_reg:
                loss = model.l1_regularize(loss, l1_lambda)
            loss.backward()
            optimizer.step()

            test_loss = (float(loss_func(out_test, test_y)))

            losses[-1] += test_loss/len(batches)
            train_losses[-1] += float(loss)/len(batches)

            print('Epoch: {:03d}:{:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, batch_no, loss, test_loss))
        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    if plot:
        size = 100
        plt.scatter(range(size), [n.detach() for n in out_test][:size], label="Predicted")
        plt.scatter(range(size), [n.detach() for n in test_y][:size], label="Correct")
        plt.title(f"Predicted and correct (first {size} subjects) - Final")
        plt.xlabel("Subject")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-pred-vs-correct-epoch-final.png")
        plt.close()

        plt.plot(losses, label="Loss")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-loss.png")
        plt.close()

        plt.plot(corrs, label="Correlation")
        plt.title("Correlation between prediction and correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-correlation.png")
        plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses}).to_csv(f"output/{test_name}-train-data.csv")
    return losses, epoch


def create_network(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1):
    # return create_network_no_conv(inp_size, out_size, conv_kernel_size, pool_size, internal_size)
    # return create_network_no_conv_pool(inp_size, out_size, conv_kernel_size, pool_size, internal_size)
    return create_network_conv(inp_size, out_size, conv_kernel_size, pool_size, internal_size)
    # return create_network_2conv(inp_size, out_size, conv_kernel_size, pool_size, internal_size)


def create_data(loader, params=None, test_case="default", plot=True):
    if params is None:
        params = {}
    filename = "SNP.csv"
    if "filename" in params:
        filename = params.pop("filename")
    if not os.path.exists(filename) and os.path.exists(pathlib.Path("csv-data")/filename):
        filename = str(pathlib.Path("csv-data")/filename)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = load_data("SNP.csv", bits=["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"])  #
    dataset = loader(filename, **params)

    if "use_dataset" in params:
        return dataset
    test_case += "-" + "-".join(f"{val}" if not isinstance(val, list) else f"{len(val)}" for val in params.values())
    save_dataset_info(dataset, test_case=test_case)
    if plot and hasattr(dataset.data, "edge_index") and dataset.data.edge_index is not None:
        plot_dataset(dataset, test_case=test_case)

    if len(dataset) > 1:
        data = [x#.to(device)
         for x in dataset]
    else:
        data = dataset[0]#.to(device)

    return data, dataset.num_features


def get_size(data):
    if hasattr(data, "x"):
        return [(data.x.shape[1], ), (1, )]
    return [(data[0].x.shape[1], ), (1, )]


def summarize(model, size, data, case_name):
    if not isinstance(model, torch.nn.Module):
        model = model()

    # summary(model, size, dtypes=[torch.FloatTensor, torch.LongTensor])
    x = data.x if hasattr(data, "x") else data[0].x
    edge_index = data.edge_index if hasattr(data, "edge_index") else data[0].edge_index
    yhat = model(x, edge_index)
    # make_dot(yhat, params=dict(list(model.named_parameters())),  show_attrs=True, show_saved=True).render(f"graphviz/{case_name}", format="png")
    # transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    # graph = hl.build_graph(model, (torch.zeros([8, 1279, 2, 40])))
    # #graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save(f"graphviz/{case_name}", format='png')
    #torch.save(model.state_dict(), "/home/lkihlman/projects/ragini/gnn/Wheat1.h5")
    # writer = SummaryWriter(f'graphviz/{case_name}')
    # writer.add_graph(model, (x, edge_index))

def get_or_create(out_size, load_when_exists=False, test_case="default", use_model_creator=False, epochs=1000, loader=load_data, trainer=train, params=None, plot=True, save_loss=True):
    if params is None:
        params = {}
    if "network" in params:
        network = params.pop("network")
    else:
        network = create_network

    if "max_no_improvement" in params:
        max_no_improvement = params.pop("max_no_improvement")
    else:
        max_no_improvement = -1
    if "learning_rate" in params:
        learning_rate = params.pop("learning_rate")
    else:
        learning_rate = 0.017

    if "weight_decay" in params:
        weight_decay = params.pop("weight_decay")
    else:
        weight_decay = 0.0

    if "l1_lambda" in params:
        l1_lambda = params.pop("l1_lambda")
    else:
        l1_lambda = 0.01

    t_kwargs = {}
    if "use_l1_reg" in params:
        t_kwargs["use_l1_reg"] = params.pop("use_l1_reg")
    if "aggregate_epochs" in params:
        t_kwargs["aggregate_epochs"] = params.pop("aggregate_epochs")
    # if "batch_size" in params:
    #     t_kwargs["batch_size"] = params.pop("batch_size")

    n_kwargs = {}
    if "device" in params:
        n_kwargs["device"] = params.pop("device")
    if "radius" in params:
        n_kwargs["radius"] = params.pop("radius")
    if "dropout" in params:
        n_kwargs["dropout"] = params.pop("dropout")
    if "add_self_loops" in params:
        n_kwargs["add_self_loops"] = params.pop("add_self_loops")
    if "conv_kernel_size" in params:
        n_kwargs["conv_kernel_size"] = params.pop("conv_kernel_size")
    if "pool_size" in params:
        n_kwargs["pool_size"] = params.pop("pool_size")
    if "internal_size" in params:
        n_kwargs["internal_size"] = params.pop("internal_size")
    if "embedding_dim" in params:
        n_kwargs["embedding_dim"] = params.pop("embedding_dim")
    if "filter_size" in params:
        n_kwargs["filter_size"] = params.pop("filter_size")
    if "filters" in params:
        n_kwargs["filters"] = params.pop("filters")

    data = create_data(loader, params, test_case=test_case, plot=plot)
    if not params.get("use_dataset", False):
        data, inp_size = data
    else:
        inp_size = 0
    path = f"model/{inp_size}-{out_size}-{test_case}.pt"
    print("Creating model")
    if use_model_creator and not (os.path.exists(path) and load_when_exists):
        model = model_creator(network, inp_size, out_size, **n_kwargs)
        optimizer = optimizer_creator(torch.optim.Adam,  lr=learning_rate)#, weight_decay=weight_decay)
    else:
        model = network(inp_size, out_size, **n_kwargs)
        optimizer = torch.optim.Adam(model.parameters(),)

    data_size = get_size(data)
    #summarize(model, data_size, data, test_case)

    if os.path.exists(path) and load_when_exists:
        model.load_state_dict(torch.load(path))
        loss_func = torch.nn.MSELoss()
        losses = []
        out_test = model((data.test.x, data.test.edge_index))
        losses.append(float(loss_func(out_test, data.test.y)))
        return model, losses, 0, data
    print("Ready to train")
    result = trainer(
        model,
        optimizer,
        data,
        test_case=test_case,
        plot=plot,
        max_no_improvement=max_no_improvement,
        l1_lambda=l1_lambda,
        epochs=epochs,
        save_loss=save_loss,
        **t_kwargs
    )
    if isinstance(result, tuple):
        result = {
            "basics": {
                "model": model,
                "losses": result[0],
                "epoch": result[1],
                "data": data
            }
        }
    else:
        if "model" not in result["basics"]:
            result["basics"]["model"] = model
        if "data" not in result["basics"]:
            result["basics"]["data"] = data
        model = result.get("basics", {}).get("model", model)

    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    return result


def all_the_things(inp: dict):

    keys = list(inp.keys())
    k = keys[0]
    values = inp[k]
    inp.pop(k)
    for v in values:
        if len(inp) == 0:
            yield {k: v}
        else:
            for result in all_the_things(inp.copy()):
                result[k] = v
                yield result


def get_info(results):
    basics = results.get("basics", {})
    model, losses, epoch, data = basics["model"], basics["losses"], basics["epoch"], basics["data"]

    if hasattr(data, "valid"):
        preds, valid_loss = validate(model, data.valid)
    else:
        valid_loss = None
        preds = None
    min_loss = min(losses)
    min_epoch = losses.index(min_loss)

    if "cors" in basics:

        print(len(basics["cors"]), min_epoch)
        #basics["cor_at_min_loss"] = basics["cors"][min_epoch]
        basics["cor_max"] = max(basics["cors"])
        # print(basics["cors"].index(basics["cor_max"]))
        basics["cor_max_epoch"] = basics["cors"].index(basics["cor_max"])
        basics["loss_at_max_cor_epoch"] = basics["cors"][basics["cor_max_epoch"]]

    return (
        model,
        losses[-1],
        min_loss,
        min_epoch,
        epoch,
        valid_loss,
        {**{k: v for k, v in basics.items() if k not in ["model", "losses", "epoch", "data", "cors"]},**{"preds":preds }}

    )


def main(cases):
    save_name = f"{time.time()}-losses-incremental-save"
    loss_dict_list = []
    for case in cases:
        epochs = case.get("epochs", 1000)
        trainer = case.get("trainer", train)
        loader = case.get("loader", load_data)
        plot = case.get("plot", True)
        use_model_creator = case.get("use_model_creator", False)
        save_loss = case.get("save_loss", True)
        for param in all_the_things(case["params"]):
            print(param)
            test_case = f"{case['test']}-{epochs}-{trainer.__name__}-{loader.__name__}-{param.get('filename', 'SNP.csv')}"
            test_builder = param.copy()
            if "bits" in test_builder:
                test_builder["bits"] = "whole" if test_builder["bits"] is None else "bits"
            if "network" in test_builder:
                test_builder["network"] = test_builder["network"].__name__
            test_builder["loader"] = loader.__name__
            use_validate = param.get("use_validation", False)
            test_case = f"{test_case}-{time.time()}"
            try:
                result = get_or_create(1, load_when_exists=False, test_case=test_case, loader=loader, epochs=epochs,
                                       trainer=trainer, use_model_creator=use_model_creator, params=param, plot=plot, save_loss=save_loss)
                if isinstance(result, tuple):
                    result = {
                        "basic": {
                            "model": result[0],
                            "losses": result[1],
                            "epoch": result[2],
                            "data": result[3]
                        }
                    }

                model, loss, min_loss, min_epoch, epoch, valid_loss, rest = get_info(result)
                loss_dict_list.append(
                    {
                        "case": case["test"],
                        "filename": param.get('filename', 'SNP.csv'),
                        "loss": loss,
                        "min_loss": min_loss,
                        "min_epoch": min_epoch,
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        **rest,
                        **test_builder
                    }
                )
                save_output(save_name, loss_dict_list[-1])
            except Exception as e:
                print(f"Failed {test_case} - {e}")

                raise e

    all_keys = set()
    for d in loss_dict_list:
        all_keys = all_keys.union(set(d.keys()))
    loss_dict_list_final = []
    empty_dict = {k: "" for k in all_keys}
    for d in loss_dict_list:
        loss_dict_list_final.append({**empty_dict, **d})

    pd.DataFrame(loss_dict_list_final, columns=list(sorted(all_keys))).to_csv(f"output/{time.time()}-losses.csv")

# WHEAT shape: 599x1280


if __name__ == "__main__":
    tests = [
        {
            "test": "timed",
            "loader": load_data_ensembles2,
            "epochs": 250,
            "trainer": train_ensemble2,
            "plot": False,
            "use_model_creator": True,
            "save_loss": False,
            "params": {
                # "filename": ["SNP.csv"],
                #"filename": ['MiceBL.csv','WHEAT_combined.csv',"QTLMASXVI.txt", 'pig4.csv'],# 'MiceBL.csv','WHEAT_combined.csv',"QTLMASXVI.txt"],
                "filename": ['MiceBL.csv','WHEAT_combined.csv',"QTLMASXVI.txt", 'pig4.csv'],
                # "split": [[(100, 200), (500, 1000), (1326, 2326), (2330, 3225)]],
                "split": [10],  # [2326], #[4071],
                "train_size": [0.7, 0.8,0.9],  # [2326], #[4071],
                # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
                #"full_split": [8],  # For ensemble2 only, a0.0027867719711243254dds the full dataset as the last enseEricsson, Hirsalantie, Kirkkonummimble
                # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
                # "split":  [(1841, 4541, 5441, 6341, 8141)],
                # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
                #"network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout, create_network_no_conv, create_network_two_no_conv, create_network_no_conv_relu, create_network_two_no_conv_relu],
                "network": [create_network_two_no_conv_relu_dropout,], #create_network_conv1D, create_network_two_no_conv_relu_dropout
                # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
                "num_neighbours": [3],
                "aggregate_epochs": [350],
                "algorithm": ['euclidean'],
                "use_weights": [True],
                "use_validation": [True],
                "smoothing": ["laplacian"],
                "separate_sets": [True,],
                "mode": ["distance"],
                "internal_size": [100],
                "max_no_improvement": [-1],
                #"filter_size": [[3,4,7]],
                #"embedding_dim": [1000],
                # "learning_rate": [0.0003851417159663891],
                # "weight_decay": [0.00001],
                # "l1_lambda": [0.3112], # [0.011inp_size200478603018649],
                #"dropout": [0.3],
                "learning_rate": [0.0027867719711243254],# 0.0020594745443455593
                "weight_decay": [0.000274339151950068],# 0.0000274339151950068
                "l1_lambda": [3.809368159814276e-05,], # 0 to hgher val#0.00001112
                "dropout": [ 0.3128021835936228], # 0 to 0.5# 0.4011713125675628
                "conv_kernel_size": [1],
                "filters": [3],
                "pool_size": [2],
                "use_l1_reg": [True],
                # "use_relu":[False]
            }
        },
        # {
        #     "test": "plain-weights",
        #     "loader": load_data,
        #     "epochs": 250,
        #     "trainer": train,
        #     "plot": True,
        #     # "use_model_creator": True,
        #     "params": {
        #         "filename": ["SNP.csv"],
        #         #"filename": ["MiceBL.csv","WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv", "SNP.csv"],
        #         # "split": [[(100, 200), (500, 1000), (1326, 2326), (2330, 3225)]],
        #         "split": [2326],  # [2326], #[4071],
        #         # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "full_split": [2326],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "algorithm": [scipy.spatial.distance.correlation, 'euclidian'],
        #         "use_weights": [True, False],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0020594745443455593],
        #         "weight_decay": [3.300385468222314e-05],
        #         "l1_lambda": [0.00001112],
        #         "dropout": [0.4011713125675628],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [False, True],
        #     }
        # },
        # {
        #     "test": "plain-weights",
        #     "loader": load_data,
        #     "epochs": 250,
        #     "trainer": train,
        #     "plot": True,
        #     # "use_model_creator": True,
        #     "params": {
        #         # "filename": ["SNP.csv"],
        #         "filename": ["MiceBL.csv","WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv"],
        #         # "split": [[(100, 200), (500, 1000), (1326, 2326), (2330, 3225)]],
        #         "split": [0.8],  # [2326], #[4071],
        #         # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "full_split": [2326],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "algorithm": [scipy.spatial.distance.correlation, 'euclidean'],
        #         "use_weights": [True, False],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0020594745443455593],
        #         "weight_decay": [3.300385468222314e-05],
        #         "l1_lambda": [0.00001112],
        #         "dropout": [0.4011713125675628],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [False, True],
        #     }
        # },
        # {
        #     "test": "ensemble-weights",
        #     "loader": load_data_ensemble2,
        #     "epochs": 250,
        #     "trainer": train_ensemble2,
        #     "plot": True,
        #     "use_model_creator": True,
        #     "params": {
        #         # "filename": ["SNP.csv"],
        #         "filename": ["MiceBL.csv","WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv"],
        #         # "split": [[(100, 200), (500, 1000), (1326, 2326), (2330, 3225)]],
        #         "split": [8],  # [2326], #[4071],
        #         # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "full_split": [2326],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "algorithm": [scipy.spatial.distance.correlation, 'euclidean'],
        #         "use_weights": [True, False],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0020594745443455593],
        #         "weight_decay": [3.300385468222314e-05],
        #         "l1_lambda": [0.00001112],
        #         "dropout": [0.4011713125675628],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [False, True],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_ensemble2,
        #     "epochs": 250,
        #     "trainer": train_none,
        #     "plot": True,
        #     "use_model_creator": True,
        #     "params": {
        #         # "filename": ["SNP.csv"],
        #         "filename": ["MiceBL.csv"],
        #         # "WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv", "SNP.csv"],
        #         # "split": [[(1841, 4541)]],
        #         "split": [8],  # [2326], #[4071],
        #         # "train_size": [0.7],  # [2326], #[4071],
        #         # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         # "network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout],
        #         "network": [create_network_two_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": [None, "Laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0020594745443455593],
        #         "weight_decay": [3.300385468222314e-05],
        #         "l1_lambda": [0.00001112],
        #         "dropout": [0.4011713125675628],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [False],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_ensemble2,
        #     "epochs": 250,
        #     "trainer": train_ensemble2,
        #     "plot": True,
        #     "use_model_creator": True,
        #     "params": {
        #         # "filename": ["SNP.csv"],
        #         "filename": ["SNP.csv"],
        #         # "split": [[(1841, 4541)]],
        #         "split": [8],  # [2326], #[4071],
        #         "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         "full_split": [2326],  # For ensemble2 only, adds the full dataset as the last ensemble
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv_dropout, create_network_two_no_conv_dropout, create_network_no_conv_relu_dropout, create_network_two_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0020594745443455593],
        #         "weight_decay": [3.300385468222314e-05],
        #         "l1_lambda": [0.00001112],
        #         "dropout": [0.4011713125675628],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [False, True],
        #     }
        # },
        # {'lr': 8.578950690515721e-05, 'weight_decay': 7.04819069164796e-05, 'optimizer': 'Adam', 'l1_lambda': 0.04468494587844092}
        # {
        #     "test": "plain-test",
        #     "loader": load_data_many,
        #     "epochs": 2500,
        #     "trainer": train,
        #     "plot": True,
        #     "params": {
        #         "filename": [["data/sim/sim_train_1.csv", "data/sim/sim_train_2.csv", "data/sim/sim_train_3.csv", "data/sim/sim_train_4.csv", "data/sim/sim_train_4.csv", "data/sim/sim_train_5.csv", "data/sim/sim_train_6.csv", "data/sim/sim_train_7.csv", "data/sim/sim_train_8.csv", "data/sim/sim_train_9.csv", "data/sim/sim_train_10.csv", "data/sim/sim_test_1.csv"]],
        #         # "split": [[(1841, 4541)]],
        #         "split": [{"test": ["data/sim/sim_test_1.csv"]}],  # [2326], #[4071],
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [8.578950690515721e-05],
        #         "weight_decay": [7.04819069164796e-05],
        #         "l1_lambda": [0.04468494587844092],
        #         # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         "remove_mean": [True],
        #         "scaled": [True],
        #         # "hot": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     # Trial 121 finished with value: 0.7487767338752747 and parameters: {'lr': 0.0001399808056488427, 'weight_decay': 7.935041364835841e-05, 'optimizer': 'Adam', 'l1_lambda': 0.10587731717470177}. Best is trial 121 with value: 0.7487767338752747
        #     "test": "plain-test",
        #     "loader": load_data,
        #     "epochs": 2000,
        #     "trainer": train,
        #     "plot": True,
        #     "params": {
        #         "filename": ["SNP.csv"],
        #         # "split": [[(1841, 4541)]],
        #         # "split": [1451],  # [2326], #[4071],
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],#
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         # "learning_rate": [8.578950690515721e-05],
        #         # "weight_decay": [7.04819069164796e-05],
        #         # "l1_lambda": [0.04468494587844092],
        #         "learning_rate": [0.01399808056488427],
        #         "weight_decay": [7.935041364835841e-05],
        #         "l1_lambda": [0.10587731717470177],
        #         # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         # "hot": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # { # Trial 121 finished with value: 0.7487767338752747 and parameters: {'lr': 0.0001399808056488427, 'weight_decay': 7.935041364835841e-05, 'optimizer': 'Adam', 'l1_lambda': 0.10587731717470177}. Best is trial 121 with value: 0.7487767338752747
        #     "test": "plain-test",
        #     "loader": load_data,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "plot": True,
        #     "params": {
        #         "filename": ["WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv"],
        #         # "split": [[(1841, 4541)]],
        #         "split": [479],  # [2326], #[4071],
        #         # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #         # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #         # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #         "network": [create_network_no_conv],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         # "learning_rate": [8.578950690515721e-05],
        #         # "weight_decay": [7.04819069164796e-05],
        #         # "l1_lambda": [0.04468494587844092],
        #         "learning_rate": [0.0007851417159663891],
        #         "weight_decay": [6.737208028206807e-05],
        #         "l1_lambda": [0.1],
        #         # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         # "hot": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #         "test": "plain-test",
        #         "loader": load_data,
        #         "epochs": 1000,
        #         "trainer": train,
        #         "plot": True,
        #         "params": {
        #             "filename": ["MiceBL.csv"],
        #             # "split": [[(1841, 4541)]],
        #             "split": [1451],  #[2326], #[4071],
        #             # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #             # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #             # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #             "network": [create_network_no_conv],
        #             # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #             "num_neighbours": [3],
        #             "smoothing": ["laplacian"],
        #             "mode": ["distance"],
        #             "internal_size": [100],
        #             "max_no_improvement": [-1],
        #             # "learning_rate": [8.578950690515721e-05],
        #             # "weight_decay": [7.04819069164796e-05],
        #             # "l1_lambda": [0.04468494587844092],
        #             "learning_rate": [0.0007851417159663891],
        #             "weight_decay": [6.737208028206807e-05],
        #             "l1_lambda": [0.1],
        #             # [0.011200478603018649],
        #             # "dropout": [0.1],
        #             "use_validation": [False],
        #             "conv_kernel_size": [32],
        #             "pool_size": [2],
        #             "use_l1_reg": [True],
        #             # "hot": [True],
        #             # "batch_size": [48],
        #             # "use_dataset": [True],
        #             # "radius": [2],
        #         }
        #  },
        # {
        #     ######################## THIS IS THE ONE YOU'RE LOOKING FOR #################################
        #     # create_network_no_conv,6.737208028206807e-05,3,71.63111877441406,100,,0.1,32,-1,False,2, ,True,distance,plain-test,864,999,0.0007851417159663891,laplacian
        #     # Loss: 68.4
        #     "test": "plain-test",
        #     "loader": load_data,
        #     "epochs": 250,
        #     "trainer": train,
        #     "plot": False,
        #     "save_loss": False,
        #     "params": {
        #         "network": [create_network_no_conv_dropout, create_network_no_conv_relu_dropout],
        #         # "bits": [["4354",  "931", "5321", "987", "1063", "5327", "5350", "5322", "5333", "942", "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [1],
        #         "split": [2326],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "algorithm": ["minkowski"],
        #         "internal_size": [1000],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0003851417159663891],
        #         "weight_decay": [0.00001],
        #         "l1_lambda": [0.3112],
        #         # [0.011200478603018649],
        #         "dropout": [0.3],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_weights": [True],
        #         "use_l1_reg": [True],
        #         # "hot": [True],
        #         "scaled": [True],
        #         "remove_mean": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #         #  "learning_rate": [0.0020594745443455593],
        #         #         "weight_decay": [3.300385468222314e-05],
        #         #         "l1_lambda": [0.00001112], # 0 to hgher val
        #     }
        # },

        # {
        #         "test": "plain-test",
        #         "loader": load_data,
        #         "epochs": 1000,
        #         "trainer": train,
        #         "plot": True,
        #         "params": {
        #             "filename": ["WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv"],
        #             # "split": [[(1841, 4541)]],
        #             "split": [479],  #[2326], #[4071],
        #             # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #             # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #             # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #             "network": [create_network_no_conv],
        #             # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #             "num_neighbours": [3],
        #             "smoothing": ["laplacian"],
        #             "mode": ["distance"],
        #             "internal_size": [100],
        #             "max_no_improvement": [-1],
        #             "learning_rate": [0.00003966710997999458],
        #             "weight_decay": [7.864464913870897e-05],
        #             "l1_lambda": [0.1],
        #             # [0.011200478603018649],
        #             # "dropout": [0.1],
        #             "use_validation": [False],
        #             "conv_kernel_size": [30],
        #             "pool_size": [4],
        #             "use_l1_reg": [False],
        #             # "hot": [True],
        #             # "batch_size": [48],
        #             # "use_dataset": [True],
        #             # "radius": [2],
        #         }
        #  },
        #  {
        #         "test": "plain-test",
        #         "loader": load_data,
        #         "epochs": 1000,
        #         "trainer": train,
        #         "plot": True,
        #         "params": {
        #             "filename": ["MiceBL.csv"],
        #             # "split": [[(1841, 4541)]],
        #             "split": [1451],  #[2326], #[4071],
        #             # "split": [(0.2, 0.4, 0.6, 0.8)], #[2326], #[4071],
        #             # "split":  [(1841, 4541, 5441, 6341, 8141)],
        #             # "split": [[(1841, 4541), (5441, 6341), (8141, 9040)]],
        #             "network": [create_network_no_conv],
        #             # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #             "num_neighbours": [3],
        #             "smoothing": ["laplacian"],
        #             "mode": ["distance"],
        #             "internal_size": [100],
        #             "max_no_improvement": [-1],
        #             "learning_rate": [0.00003966710997999458],
        #             "weight_decay": [7.864464913870897e-05],
        #             "l1_lambda": [0.1],
        #             # [0.011200478603018649],
        #             # "dropout": [0.1],
        #             "use_validation": [False],
        #             "conv_kernel_size": [30],
        #             "pool_size": [4],
        #             "use_l1_reg": [False],
        #             # "hot": [True],
        #             # "batch_size": [48],
        #             # "use_dataset": [True],
        #             # "radius": [2],
        #         }
        #  },
         # {
         #        "test": "plain-test",
         #        "loader": load_data_hot2,
         #        "epochs": 1000,
         #        "trainer": train,
         #        "params": {
         #            "network": [create_network_no_conv],
         #            # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
         #            "num_neighbours": [3],
         #            "smoothing": ["laplacian"],
         #            "mode": ["distance"],
         #            "internal_size": [100],
         #            "max_no_improvement": [-1],
         #            "learning_rate": [0.0009975243792625852],
         #            "weight_decay": [5.5225422907453564e-05],
         #            "l1_lambda": [0.0010137329373397902],
         #            # [0.011200478603018649],
         #            # "dropout": [0.1],
         #            "use_validation": [False],
         #            "conv_kernel_size": [31],
         #            "pool_size": [2],
         #            "use_l1_reg": [True],
         #            # "batch_size": [48],
         #            # "use_dataset": [True],
         #            # "radius": [2],
         #        }
         # },
        #  {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0003966710997999458, 0.0007851417159663891, 0.000201033951607494779, 0.0009997463955364163],
        #         "weight_decay": [7.864464913870897e-05, 6.737208028206807e-05, 0, 1.3064967052435184e-05],
        #         "l1_lambda": [0.02130714695513579, 0.017376699481646562, 0.1, 0.0010614983173646143],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0009997463955364163],
        #         "weight_decay": [1.3064967052435184e-05],
        #         "l1_lambda": [0.0010614983173646143],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [33],
        #         "pool_size": [8],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        #  {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0003966710997999458, 0.0007851417159663891, 0.000201033951607494779],
        #         "weight_decay": [7.864464913870897e-05, 6.737208028206807e-05, 0],
        #         "l1_lambda": [0.02130714695513579, 0.017376699481646562, 0.1],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0003966710997999458],
        #         "weight_decay": [7.864464913870897e-05],
        #         "l1_lambda": [0.02130714695513579],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [32],
        #         "pool_size": [2],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.0007851417159663891],
        #         "weight_decay": [ 6.737208028206807e-05],
        #         "l1_lambda": [0.017376699481646562],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [38],
        #         "pool_size": [5],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.000201033951607494779],
        #         "weight_decay": [0],
        #         "l1_lambda": [0.1],  # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [30],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         # "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.000201033951607494779],
        #         "weight_decay": [0],
        #         "l1_lambda": [0.1], # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [30],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         #"radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2_single,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.000201033951607494779],
        #         "weight_decay": [0],
        #         "l1_lambda": [0.1], # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [30],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         #"radius": [2],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_hot2_hop_single,
        #     "epochs": 1000,
        #     "trainer": train,
        #     "params": {
        #         "network": [create_network_k_hop],
        #         # "bits": [[str(x) for x in range(1, 250)]],
        #         # "bits": [["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.000201033951607494779],
        #         "weight_decay": [0],
        #         "l1_lambda": [0.1], # [0.011200478603018649],
        #         # "dropout": [0.1],
        #         "use_validation": [False],
        #         "conv_kernel_size": [30],
        #         "use_l1_reg": [True],
        #         # "batch_size": [48],
        #         # "use_dataset": [True],
        #         #"radius": [2],
        #     }
        # },
        # { ## Loss: 82
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "params": {
        #         "network": [create_network_no_conv_relu_dropout, create_network_no_conv_dropout],
        #         "bits": [None],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.010760757204612474, 0.012281111982078345, 0.009033951607494779],
        #         "weight_decay": [8.742904779816756e-05, 0.0003714684523938825, 0.0006854732845832472],
        #         "l1_lambda": [0.012057241336328557, 0.010008349017405088, 0.011200478603018649],
        #         "use_validation": [False],
        #         #"radius": [2],
        #     }
        # },
        # {  # Loss: 83
        #     "test": "plain-test",
        #     "loader": load_data_hot2,
        #     "params": {
        #         "network": [create_network_no_conv],
        #         "bits": [None],
        #         "num_neighbours": [3],
        #         "smoothing": ["laplacian"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "learning_rate": [0.009354062456990608, 0.01189334272879214, 0.010760757204612474],
        #         "weight_decay": [0, 8.742904779816756e-05, ],
        #         "l1_lambda": [0.005984175366619979, 0.012871823398051763, 0.012057241336328557],
        #         "use_validation": [False],
        #         #"radius": [2],
        #     }
        # },
        #{
            #"test": "plain-test",
            #"loader": load_data_single,
            #"params": {
                #"network": [create_network_conv, create_network_no_conv],
                #"bits": [None],
                #"num_neighbours": [5, 10, 20, 35, 50],
                #"smoothing": ["laplacian", "none"],
                #"mode": ["connectivity", "distance"],
                #"internal_size": [100],
                #"max_no_improvement": [-1],
                #"use_validation": [False],
            #}
        #},
        # { # Epoch: 454, Loss: 14.95922, Test Loss: 97.60881
        #     "test": "plain-test",
        #     "loader": load_data_hot2_hop,
        #     "params": {
        #         "network": [create_network_k_hop],
        #         "bits": [None],
        #         "num_neighbours": [35],
        #         "smoothing": ["laplacian"],
        #         "mode": ["connectivity"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "use_validation": [False],
        #         "radius": [3],
        #     }
        # },
        # {
        #     "test": "plain-test",
        #     "loader": load_data_two,
        #     "params": {
        #         "network": [create_network_conv_two_diff],
        #         "bits": [None],
        #         "num_neighbours": [5],
        #         "smoothing": ["none"],
        #         "mode": ["distance"],
        #         "internal_size": [100],
        #         "max_no_improvement": [-1],
        #         "use_validation": [False],
        #         "radius": [2],
        #     }
        # },
        # {
        #     "test": "plain",
        #     "loader": load_data,
        #     "params": {
        #         "bits": [None, ["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]],
        #         "num_neighbours": [1, 2, 5, 10, 15, 20],
        #         "smoothing": ["laplacian", "none"],
        #         "mode": ["connectivity", "distance"]
        #     }
        # },
        # {
        #     "test": "lapstruct",
        #     "loader": load_data_lapstruct,
        #     "params": {
        #         "bits": [None, ["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"]]
        #         "num_neighbours": [1, 2, 3, 4, 5, 10, 15, 20, 30, 50],
        #         "smoothing": ["laplacian", "none"],
        #         "mode": ["connectivity", "distance"],
        #         "num_vars": [2, 3, 4, 5]
        #     }
        # },
        # {
        #     "test": "amat",
        #     "loader": load_data_amat,
        #     "params": {
        #         "num_neighbours": [1, 2, 5, 10, 15, 20],
        #         "smoothing": ["laplacian"],
        #         "mode": ["connectivity", "distance"],
        #     }
        # },
        # {
        #     "test": "radius",
        #     "loader": load_data_radius,
        #     "params": {
        #         "bits": [None, ["4354",  "931", "5321", "987", "1063", "5327", "5350", "5322", "5333", "942", "1014", "923", "1030", "1106", "979"]],
        #         "num_radius": [50, 75, 100, 150],
        #         "smoothing": ["laplacian", "none"],
        #         "mode": ["connectivity", "distance"],
        #         "metric": ["minkowski", "manhattan", "chebyshev"],
        #     }
        # },
    ]
    main(tests)
