
import os
import pathlib
import pickle
import shutil
import time

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

EPOCHS = 250

def train(model_cons, data, optimizer_cons,  test_case="default", plot=False, epochs=300, max_no_improvement=-1, l1_lambda=0.01, use_l1_reg=False, batch_size=0, save_loss=True, **kwargs):
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

    l1_lambda = kwargs.pop('l1_lambda') if "l1_lambda" in kwargs else False
    #data.to(device)
    for epoch in range(epochs):
        model = model_cons(**kwargs)
        optimizer = optimizer_cons(model, **kwargs)
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            arg_tuple = data.x.float(), data.edge_index, data.edge_weight.float()
        else:
            arg_tuple = data.x, data.edge_index

        out = model(arg_tuple) #.to(device)

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

def plain_trainer_no_edge(model_cons, data, optimizer_cons, loss_func, **kwargs):
    no_improvement_cap = -1
    no_improvement = 0
    test_loss = 100000000
    last_loss = test_loss
    test_name = f"{time.time()}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l1_lambda = kwargs.pop('l1_lambda') if "l1_lambda" in kwargs else False

    model = model_cons(**kwargs)
    optimizer = optimizer_cons(model, **kwargs)
    best_so_far = save(model, test_name)
    if not isinstance(data, list):
        data = [data]
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for batch_no, datum in enumerate(data):
            out = model(datum.x)
            model.eval()
            if len(datum.test.x) > 0:
                out_test = model(datum.test.x)
                test_y = datum.test.y
            else:
                out_test = model(datum.x)
                test_y = datum.y
            # out = model(data.x, data.edge_index)
            # out_test = model(data.test.x, data.test.edge_index)
            loss = loss_func(out, datum.y)
            #
            if l1_lambda is not False:
                l1_parameters = []
                for parameter in model.parameters():
                    l1_parameters.append(parameter.view(-1))
                loss = loss + l1_lambda * torch.abs(torch.cat(l1_parameters)).sum()

            loss.backward()
            optimizer.step()
            test_loss = (float(loss_func(out_test, test_y)))

            if test_loss < last_loss:
                no_improvement = 0
                delete(best_so_far)
                best_so_far = save(model, test_name)
            else:
                no_improvement += 1
            last_loss = test_loss

    model = load(best_so_far)

    model.eval()
    if len(datum.test.x) > 0:
        out_test = model(datum.test.x)
        test_y = datum.test.y
    else:
        out_test = model(datum.x)
        test_y = datum.y
    test_loss = (float(loss_func(out_test, test_y)))
    delete(best_so_far)

    return -test_loss



if __name__ == "__main__":
    dataset = create_data(load_data, params={'filename': "csv-data/QTLMASXVI.txt"})
    data, inp_size = dataset
    out_size = 1

    # model_cons = lambda conv_kernel_size=None, filters=None, **_: (
    #     create_network_conv1D(inp_size, out_size, conv_kernel_size=int(conv_kernel_size), filters=int(filters))
    # )
    model_cons = lambda conv_kernel_size=None, filters=None, **_: (
        create_network_two_no_conv_relu_dropout(inp_size, out_size)
    )
    optimizer_cons = lambda x, lr, **_: torch.optim.Adam(x.parameters(), lr=abs(lr))# , weight_decay=weight_decay)

    loss_func = torch.nn.MSELoss()
    fit_with_partial = lambda *args, **kwargs: train(model_cons, data, optimizer_cons, loss_func, *args, **kwargs)

    bounds_2 = {
        'lr': ( 0.0025, 0.0055),
        'l1_lambda': (0.01, 0.05),
        #'filters': (20,100),
        #'conv_kernel_size': (10, 50),
    }
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.9)

    optimizer = BayesianOptimization(
        f            = fit_with_partial,
        pbounds      = bounds_2,
        verbose      = 1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state = 1,
        bounds_transformer = bounds_transformer
    )


    optimizer.maximize(init_points = 5, n_iter = 10,)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))


    print(optimizer.max)


