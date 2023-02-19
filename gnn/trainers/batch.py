import time

import torch
import pandas as pd

from gnn.trainers.util import l1_regularize


def train_batch(
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

    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses, "corr": corrs}).to_csv(
            f"output/{test_name}-train-data.csv")
    return losses, epoch
