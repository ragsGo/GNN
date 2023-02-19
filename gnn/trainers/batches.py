import time

import pandas as pd
import torch


def train_batches(
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
    # corrs = []
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

    if save_loss:
        pd.DataFrame({"train_loss": train_losses, "loss": losses}).to_csv(f"output/{test_name}-train-data.csv")
    return losses, epoch

