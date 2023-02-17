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
from sklearn.metrics import silhouette_score
from torch.nn import Identity
import seaborn as sns
from torchsummary import summary
# from torchviz import make_dot
# import h5py
from load import load_data

import scipy
import gc

gc.collect()
from conv1D_network import create_network_conv1D


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
        plt.title("Correlation between prediction and correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/{test_name}-gnn-correlation.png")
        plt.close()
    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
            f"output/{test_name}-train-data.csv")
    return  {
                "basics": {
                    "losses": losses,
                    "epoch": epochs,
                    "cors": corrs,
                }
    }
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

def get_or_create(out_size, load_when_exists=False, test_case="default", use_model_creator=False, epochs=1000, loader=load_data, trainer=train_no_edge, params=None, plot=True, save_loss=True):
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
def l1_regularize(model, loss, l1_lambda):
    if hasattr(model, "l1_regularize"):
        return model.l1_regularize(loss, l1_lambda)
    l1_parameters = []
    for parameter in model.parameters():
        l1_parameters.append(parameter.view(-1))

    l1 = l1_lambda * torch.abs(torch.cat(l1_parameters)).sum().float()

    return loss + l1

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
        basics["cor_at_min_loss"] = basics["cors"][min_epoch]
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



def main(cases):
    save_name = f"{time.time()}-losses-incremental-save"
    loss_dict_list = []
    for case in cases:
        epochs = case.get("epochs", 1000)
        trainer = case.get("trainer", train_no_edge)
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

if __name__ == "__main__":
    tests = [
        {
            "test": "timed",
            "loader": load_data,
            "epochs": 250,
            "trainer": train_no_edge,
            "plot": False,
            #"use_model_creator": True,
            "save_loss": False,
            "params": {
               "filename": ['WHEAT_combined.csv',],
                "split": [0.8],  # [2326], #[4071],
                "network": [create_network_conv1D,],
                "num_neighbours": [3],
                "aggregate_epochs": [350],
                "algorithm": ['euclidean'],
                "use_weights": [True],
                "use_validation": [False],
                "smoothing": ["laplacian"],
                "mode": ["distance"],
                "internal_size": [100],
                "max_no_improvement": [-1],
                "learning_rate": [0.0025],# 0.0020594745443455593
                "weight_decay": [0.000274339151950068],# 0.0000274339151950068
                "l1_lambda": [0.00001112,], # 0 to hgher val#0.00001112
                #"dropout": [0.4011713125675628], # 0 to 0.5# 0.4011713125675628
                "conv_kernel_size": [2],
                "filters": [6],
                "pool_size": [2],
                "use_l1_reg": [True],
            }
        },
    ]
    main(tests)
