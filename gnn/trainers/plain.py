import time

import numpy as np
import pandas as pd
import torch


def train(
        model,
        optimizer,
        data,
        test_case="default",
        epochs=1000,
        max_no_improvement=-1,
        l1_lambda=0.01,
        use_l1_reg=False,
        save_loss=True,
        **_
):
    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
        optimizer = optimizer(model)

    loss_func = torch.nn.MSELoss()
    losses = []
    train_losses = []
    corrs = []
    no_improvement = 0
    test_loss = 100000000
    least_loss = test_loss
    test_name = f"{time.time()}-{test_case}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    for epoch in range(epochs):
        model.to(device).train()

        optimizer.zero_grad()
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            arg_tuple = data.x.float(), data.edge_index, data.edge_weight.float()
        else:
            arg_tuple = data.x, data.edge_index

        out = model(arg_tuple).to(device)
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

        corr = np.corrcoef([n.detach().squeeze().cpu().numpy() for n in out_test],
                           [n.detach().squeeze().cpu().numpy() for n in test_y])

        corrs.append(corr[0][1])

        loss = loss_func(out, y)
        if use_l1_reg:
            loss = model.l1_regularize(loss, l1_lambda)
        loss.backward()
        optimizer.step()

        test_loss = (float(loss_func(out_test, test_y)))

        losses.append(test_loss)
        train_losses.append(float(loss))
        print('Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

        if losses[-1] < least_loss:
            no_improvement = 0
            least_loss = losses[-1]
        else:
            no_improvement += 1

        if 0 < max_no_improvement <= no_improvement:
            break

    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
            f"output/{test_name}-train-data.csv"
        )
    return {
            "basics": {
                "model": model,
                "losses": losses,
                "epoch": epochs,
                "cors": corrs,
            }
    }
