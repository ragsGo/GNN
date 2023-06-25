import os
import pathlib

import time
import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

from gnn.helpers.forestfire import forest_fire
from gnn.loaders.load import load_data
from gnn.loaders.util import split_dataset_graph, naive_partition
from gnn.networks.networks import create_network_conv, create_network_two_no_conv_relu_dropout, \
    create_network_two_no_conv_dropout, create_network_no_conv_dropout, create_network_two_relu, \
    create_network_no_conv_relu_dropout, create_network_no_relu
from gnn.trainers.ensemble2 import train_ensemble2
from gnn.trainers.plain import train


def model_creator(model_type, *args, **kwargs):
    def creator():
        return model_type(*args, **kwargs)
    return creator


def optimizer_creator(opt_type, *args, **kwargs):
    def creator(model):
        return opt_type(model.parameters(), *args, **kwargs)
    return creator


def save_output(name, output, directory="output"):
    ext = ".csv" if name[-4:] != ".csv" else ""
    name = f"{directory}/{name}{ext}"
    keys = sorted(output.keys())
    out_df = pd.DataFrame([output])
    if os.path.exists(name):
        existing_df = pd.read_csv(name)
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


def get_edges(dataset):
    edges_raw = dataset.edge_index[0][0] if isinstance(dataset.edge_index, (tuple, list)) else dataset.edge_index
    edges_raw = edges_raw.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    return edges


def save_dataset_info(datasets, test_case="default"):
    data_info = []
    for no, dataset in enumerate(datasets):
        if not hasattr(dataset, "edge_index"):
            continue
        edges = get_edges(dataset)

        if hasattr(dataset, "test") and dataset.test is not None:
            y = dataset.test.y
        else:
            y = dataset.y
        g = nx.Graph()
        g.add_edges_from(edges)
        degree = [x[1] for x in nx.degree(g)]
        if hasattr(dataset, "edge_weight") and dataset.edge_weight is not None:
            weights = dataset.edge_weight
        else:
            weights = [0]
        average_y = float(sum(y)/len(y))
        min_y = float(min(y))
        max_y = float(max(y))
        data_info.append({
            "dataset": f"{test_case}-{no}",
            "nodes": len(g.nodes),
            "edges": len(g.edges),
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

    if hasattr(datasets, "valid") and hasattr(datasets.valid, "edge_index"):
        edges = get_edges(datasets.valid)

        g = nx.Graph()
        g.add_edges_from(edges)
        degree = [x[1] for x in nx.degree(g)]

        if hasattr(datasets.valid, "edge_weight") and datasets.valid.edge_weight is not None:
            weights = datasets.valid.edge_weight
        else:
            weights = [0]
        y = datasets.valid.y
        average_y = float(sum(y)/len(y))
        min_y = float(min(y))
        max_y = float(max(y))
        data_info.append({
            "dataset": f"{test_case}-validation",
            "nodes": len(g.nodes),
            "edges": len(g.edges),
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


def draw_edges(edges, save_as="plot.png", fig_size=(55, 50), with_labels=True):
    g = nx.Graph()
    g.add_edges_from(edges)
    plt.rcParams["figure.figsize"] = fig_size
    nx.draw(g, with_labels=with_labels)

    plt.savefig(save_as)
    plt.close()


def plot_dataset(datasets, test_case="default"):
    test_data = None
    for no, dataset in enumerate(datasets):
        if not hasattr(dataset, "edge_index"):
            continue
        edges = get_edges(dataset)
        draw_edges(edges, f"images/{test_case}-{no}-neighbourhood_graph.png")
        if test_data is None or test_case is not dataset.test:
            test_data = dataset.test
            edges = get_edges(test_data)
            draw_edges(edges, f"images/{test_case}-test-neighbourhood_graph.png")


    if hasattr(datasets, "valid") and hasattr(datasets.valid[0], "edge_index"):
        edges = get_edges(datasets.valid[0])
        draw_edges(edges, f"images/{test_case}-validation-neighbourhood_graph.png")


def validate(model, data, loss_func=torch.nn.MSELoss()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pred = model(data["x"], data["edge_index"])
    loss = loss_func(pred, data["y"])
    return pred, float(loss)


def create_network(inp_size, out_size, conv_kernel_size=30, pool_size=2, internal_size=-1, **_):
    return create_network_conv(inp_size, out_size, conv_kernel_size, pool_size, internal_size)


def create_data(loader, params=None, test_case="default", plot=True):
    if params is None:
        params = {}
    filename = "SNP.csv"
    if "filename" in params:
        filename = params.pop("filename")
    if not os.path.exists(filename) and os.path.exists(pathlib.Path("csv-data") / filename):
        filename = str(pathlib.Path("csv-data") / filename)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = loader(filename, **params)

    if "use_dataset" in params:
        return dataset

    #test_case += "-" + "-".join(f"{val}" if not isinstance(val, list) else f"{len(val)}" for val in params.values())
    # test_case = "todays"
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
    # x = data.x if hasattr(data, "x") else data[0].x
    # edge_index = data.edge_index if hasattr(data, "edge_index") else data[0].edge_index
    # yhat = model(x, edge_index)
    # make_dot(yhat, params=dict(list(model.named_parameters())),  show_attrs=True, show_saved=True)
    # .render(f"graphviz/{case_name}", format="png")
    # transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    # graph = hl.build_graph(model, (torch.zeros([8, 1279, 2, 40])))
    # #graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save(f"graphviz/{case_name}", format='png')
    # torch.save(model.state_dict(), "/home/lkihlman/projects/ragini/gnn/Wheat1.h5")
    # writer = SummaryWriter(f'graphviz/{case_name}')
    # writer.add_graph(model, (x, edge_index))


def get_or_create(
        out_size,
        load_when_exists=False,
        test_case="default",
        use_model_creator=False,
        epochs=1000,
        loader=load_data,
        trainer=train,
        params=None,
        plot=True,
        save_loss=True
):
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
    if os.path.exists(path) and load_when_exists:
        model = network(inp_size, out_size, **n_kwargs)
        model.load_state_dict(torch.load(path))
        loss_func = torch.nn.MSELoss()
        losses = []
        out_test = model((data.test.x, data.test.edge_index))
        losses.append(float(loss_func(out_test, data.test.y)))
        return {
            "basics": {
                "model": model,
                "losses": losses,
                "epoch": 0,
                "data": data
            }
        }

    # if use_model_creator and not (os.path.exists(path) and load_when_exists):
    model = model_creator(network, inp_size, out_size, **n_kwargs)
    optimizer = optimizer_creator(torch.optim.Adam,  lr=learning_rate)  # , weight_decay=weight_decay)
    # else:
    #     model = network(inp_size, out_size, **n_kwargs)
    #     optimizer = torch.optim.Adam(model.parameters(),)

    # data_size = get_size(data)
    # summarize(model, data_size, data, test_case)


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
        # basics["cor_at_min_loss"] = basics["cors"][min_epoch]
        basics["cor_max"] = max(basics["cors"])
        basics["cor_max_epoch"] = basics["cors"].index(basics["cor_max"])
        basics["loss_at_max_cor_epoch"] = basics["cors"][basics["cor_max_epoch"]]

    if "dcors" in basics:
        # basics["cor_at_min_loss"] = basics["cors"][min_epoch]
        basics["dcor_max"] = max(basics["dcors"])
        basics["dcor_max_epoch"] = basics["dcors"].index(basics["dcor_max"])
        basics["loss_at_max_dcor_epoch"] = basics["dcors"][basics["dcor_max_epoch"]]

    return (
        model,
        losses[-1],
        min_loss,
        min_epoch,
        epoch,
        valid_loss,
        {
            **{k: v for k, v in basics.items() if k not in ["model", "losses", "epoch", "data", "cors"]},
            # **{"preds": preds},
        },
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
            #print(param)
            test_case = f"{case['test']}-{epochs}-{trainer.__name__}-" \
                        f"{loader.__name__}-{param.get('filename', 'SNP.csv')}"
            test_builder = param.copy()
            if "bits" in test_builder:
                test_builder["bits"] = "whole" if test_builder["bits"] is None else "bits"
            if "network" in test_builder:
                test_builder["network"] = test_builder["network"].__name__
            test_builder["loader"] = loader.__name__
            # use_validate = param.get("use_validation", False)
            test_case = f"{test_case}-{time.time()}"
            try:
                result = get_or_create(
                    1,
                    load_when_exists=False,
                    test_case=test_case,
                    loader=loader,
                    epochs=epochs,
                    trainer=trainer,
                    use_model_creator=use_model_creator,
                    params=param,
                    plot=plot,
                    save_loss=save_loss
                )
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
            "loader": load_data,
            "epochs": 650,
            "trainer": train_ensemble2,
            "plot": True,
            "use_model_creator": True,
            "save_loss": False,
            "params": {

                "filename": ['pig4.csv'], #'WHEAT_combined.csv', 'MiceBL.csv', 'QTLMASXVI.txt', 'pig4.csv', 'SNP.csv'],# 'MiceBL.csv' ,  'QTLMASXVI.txt', 'pig4.csv'],
                "split": [0.8],  # [2326], #[4071],
                # "train_size": [0.7, 0.8,model_cons 0.9],  # [2326], #[4071],
                # "add_full": [True],  # For ensemble2 only, adds the full dataset as the last ensemble
                # "full_split": [8],  # For ensemble2 only, adds the full dataset as the last ensemble
                "network": [create_network_two_no_conv_dropout], #, create_network_two_no_conv_dropout, create_network_no_conv_dropout, create_network_two_relu, create_network_no_conv_relu_dropout, create_network_no_relu],
                # "bits": [[
                #    "4354",
                #    "931",
                #    "5321",
                #    "987",
                #    "1063",
                #    "5327",
                #    "5350""partition": forest_fire},,
                #    "5322",
                #    "5333",
                #    "942",
                #    "1014",
                #    "923",
                #    "1030",
                #    "1106",
                #    "979"
                # ]],
                "split_algorithm": [split_dataset_graph],
                "split_algorithm_params": [ {"partition": naive_partition}],
                "num_neighbours": [3, 5, 8, 10],
                "aggregate_epochs": [350],
                "algorithm": ['euclidean'],
                "batches": [3, 5, 8, 10],
                "use_weights": [True],
                "use_validation": [True],
                "smoothing": ["laplacian"],
                "separate_sets": [True],
                "mode": ["distance"],
                "internal_size": [100],
                "max_no_improvement": [-1],
                "learning_rate": [ 0.0020594745443455593,],# 0.002059474544345559, 0.000020594745443455593],  # 0.0020594745443455593
                "weight_decay": [0],  # 0.0000274339151950068
                "l1_lambda": [0.0001112],  # 0 to hgher val#0.00001112
                "dropout": [0.4011713125675628],  # 0 to 0.5# 0.4011713125675628
                #"conv_kernel_size": [1],
                #"filters": [3],
                "pool_size": [2],
                "use_l1_reg": [ False],
            }
        },
    ]
    main(tests)
