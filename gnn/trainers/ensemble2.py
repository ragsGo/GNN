import pathlib
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import Identity

from .util import get_args_tuple, l1_regularize
from ..networks.networks import Ensemble


def train_ensemble2(
        model,
        optimizer,
        data,
        test_case="default",
        epochs=1000,
        aggregate_epochs=None,
        max_no_improvement=-1,
        l1_lambda=0.01,
        use_l1_reg=False,
        save_loss=True,
        early_stopper=lambda _, __, ___: None,
        print_epochs=True,
        **_
):
    test_name = f"{time.time()}-{test_case}"
    if aggregate_epochs is None:
        aggregate_epochs = epochs
    orig_model = model
    orig_optimizer = optimizer
    paths = []
    losses_dict = {}
    pathlib.Path(f"model/ensemble/{test_name}/").mkdir(parents=True, exist_ok=True)
    min_loss_so_far = 1000000000
    best_model_so_far = None
    for i_batch, sample_batched in enumerate(data[:-1]):
        model = orig_model()
        optimizer = orig_optimizer(model)
        loss_func = torch.nn.MSELoss()
        no_improvement = 0
        losses = []
        train_losses = []
        corrs = []
        test_loss = 100000000
        least_loss = test_loss
        path = f"model/ensemble/{test_name}/{i_batch}.pt"
        paths.append(path)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            args_tuple = get_args_tuple(sample_batched)

            out = model(*args_tuple)

            model.eval()
            y = sample_batched.y

            test_tuple = get_args_tuple(sample_batched.test)

            out_test = model(*test_tuple)

            test_y = sample_batched.test.y
            pred_val = out_test.detach().squeeze().numpy()
            true_val = test_y.detach().squeeze().numpy()
            df = pd.DataFrame(data=np.column_stack((true_val, pred_val)), columns=['true_val', 'pred_val'])
            df["true_val_r"] = df.true_val.rank()
            df["pred_val_r"] = df.pred_val.rank()

            corr = np.corrcoef(
                [n.detach().squeeze().numpy() for n in out_test],
                [n.detach().squeeze().numpy() for n in test_y]
            )

            corr = corr - np.min(corr) / (np.max(corr) - np.min(corr))

            corrs.append(corr[0][1])

            loss = loss_func(out, y)

            if use_l1_reg:
                loss = l1_regularize(model, loss, l1_lambda)

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))
            test_loss = (float(loss_func(out_test, test_y)))
            losses.append(test_loss)
            if test_loss < least_loss:
                no_improvement = 0
                least_loss = losses[-1]
                torch.save(model.state_dict(), path)
            else:
                no_improvement += 1

            if test_loss < min_loss_so_far:
                min_loss_so_far = test_loss
                best_model_so_far = model

            if 0 < max_no_improvement <= no_improvement:
                break
            if print_epochs:
                print(
                    'Batch: {:02d}, Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(
                        i_batch,
                        epoch,
                        train_losses[-1],
                        losses[-1]
                    )
                )

        def cleanup():
            shutil.rmtree(f"model/ensemble/{test_name}", ignore_errors=True)
        early_stopper(min(losses), i_batch, cleanup)

        losses_dict[i_batch] = losses

        if save_loss:
            pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
                f"output/{test_name}-train-ensemble2-{i_batch}-data.csv"
            )
    final_outs = []
    loss_func = torch.nn.MSELoss()

    losses_and_epochs = {}
    for k, k_losses in losses_dict.items():
        losses_and_epochs[f"min_loss_{k}"] = min(k_losses)
        losses_and_epochs[f"min_epoch_{k}"] = k_losses.index(losses_and_epochs[f"min_loss_{k}"])

    valid_tuple = get_args_tuple(data[-1])

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

        if test_loss < least_loss:
            least_loss = test_loss
            torch.save(aggr_model.state_dict(), aggr_path)

        loss.backward()
        optimizer.step()
        aggr_losses.append(float(test_loss))

        corr = np.corrcoef([n.detach().squeeze().numpy() for n in out_test],
                           [n.detach().squeeze().numpy() for n in test_y])
        corrs.append(corr[0][1])

        if print_epochs:
            print(
                'Batch: aggregation, Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(
                    aggr_epoch,
                    loss,
                    test_loss
                )
            )

    aggr_model.load_state_dict(torch.load(aggr_path))
    torch.save(aggr_model, aggr_path)

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
    corr = np.corrcoef(
        [n.detach().squeeze().numpy() for n in pred_val],
        [n.detach().numpy() for n in data[-1].y.flatten()]
    )
    best_corr = (corr[0][1])

    var_phen = np.cov(data[-1].y.flatten())
    err = pred_val - data[-1].y.flatten()
    var_err = np.cov(err.detach().numpy())
    var_gen = var_phen - var_err
    her = var_gen / var_phen

    shutil.rmtree(f"model/ensemble/{test_name}", ignore_errors=True)
    return {
            "basics": {
                "losses": list(sorted(losses_dict.values(), key=lambda x: min(x))[-1]),
                "epoch": epochs,
                #"all_losses": sum(losses_dict.values(), [loss]),
                "best_loss": best_loss,
                "aggr_loss": aggr_loss,
                "min_loss_validation": loss,
                "model": best_model_so_far,
                "aggr_model": aggr_model,
                "cors": corrs,
                "cor_best": best_corr,
                "cor_valid": avg_corr,
                "heritability": np.mean(her),
                **losses_and_epochs
            }
    }
