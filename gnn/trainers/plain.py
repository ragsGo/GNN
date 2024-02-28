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
        epoch_callback=lambda epoch, model, loss: None,
        batch_callback=lambda batch, model, loss: None,
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
    if not isinstance(data, list):
        data = [data]
    test_loss = 0
    batched = len(data) > 1
    for idx, datum in enumerate(data):
        datum.to(device)
        for epoch in range(epochs):
            model.to(device).train()

            optimizer.zero_grad()
            if hasattr(datum, "edge_weight") and datum.edge_weight is not None:
                arg_tuple = datum.x.float(), datum.edge_index, datum.edge_weight.float()
            else:
                arg_tuple = datum.x, datum.edge_index

            out = model(arg_tuple).to(device)
            model.eval()

            if not hasattr(datum, "range"):
                y = datum.y
                if len(datum.test.x) > 0:
                    if hasattr(datum.test, "edge_weight") and datum.test.edge_weight is not None:
                        test_tuple = datum.test.x.float(), datum.test.edge_index, datum.test.edge_weight.float()
                    else:
                        test_tuple = datum.test.x, datum.test.edge_index
                    out_test = model(test_tuple)
                    test_y = datum.test.y
                else:
                    if hasattr(datum, "edge_weight") and datum.edge_weight is not None:
                        test_tuple = datum.x, datum.edge_index, datum.edge_weight
                    else:
                        test_tuple = datum.x, datum.edge_index
                    out_test = model(test_tuple)
                    test_y = datum.y
            else:
                out_test = out[datum.test.range.start: datum.test.range.stop]
                out = out[datum.range.start: datum.range.stop]
                y = datum.y[datum.range.start: datum.range.stop]
                test_y = datum.y[datum.test.range.start: datum.test.range.stop]

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
            batch_string = "Batch: {:03d}, ".format(idx) if batched else ""
            print(batch_string + 'Epoch: {:03d}, Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, train_losses[-1], losses[-1]))

            if losses[-1] < least_loss:
                no_improvement = 0
                least_loss = losses[-1]
            else:
                no_improvement += 1
            modified_model = epoch_callback(epoch, model, float(test_loss))
            if modified_model is not None:
                model = modified_model
            if 0 < max_no_improvement <= no_improvement:
                break

        modified_model = batch_callback(idx, model, float(test_loss))
        if modified_model is not None:
            model = modified_model
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
