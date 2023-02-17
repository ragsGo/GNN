from random import random

from torch.autograd import Variable
from torch.nn import ReLU, Linear, Sequential, Conv1d, Flatten, MaxPool1d
import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def train(loader, plot=False):
    train_accuracies, test_accuracies = list(), list()
    loss_func = torch.nn.MSELoss()
    losses = []
    corrs = []
    for epoch in range(250):
        outs = []
        individual_losses = []
        individual_corrs = []
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            # for i, inp in enumerate(data):
            #     if data.train_mask[i]:
            # model.train()

            b_x = Variable(batch_x.unsqueeze(1).to(device))
            b_y = Variable(batch_y.to(device))

            # inp = data
            # print(inp)
            optimizer.zero_grad()
            out = model(b_x)
            outs.append(out)
            loss = loss_func(out, b_y)
            loss.backward()
            individual_losses.append(float(loss))
            optimizer.step()
            corr = np.corrcoef([n[0] for n in out.detach().numpy()], batch_y.data.detach().numpy())
            individual_corrs.append(corr[0][1])
            if plot and epoch == 1 and step == 1:
                plt.scatter(range(len(out)), [n[0] for n in out.detach().numpy()], label="Predicted")
                plt.scatter(range(len(b_y)), [n.detach() for n in b_y], label="Correct")
                plt.title("Predicted and correct (first 100 subjects) - Epoch 0")
                plt.xlabel("Subject")
                plt.ylabel("Value")
                plt.legend(loc='upper right')
                plt.savefig(f"images/cnn-1-pred-vs-correct-epoch-0-{test_case}.png")
                plt.close()
            if step == 1:
                first_batch_out = out
                first_batch_y = b_y

        losses.append(sum(individual_losses)/len(individual_losses))
        corrs.append(sum(individual_corrs)/len(individual_corrs))

        # train_accuracies.append(train_acc)
        # test_accuracies.append(test_acc)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, losses[-1]))

    if plot:
        plt.scatter(range(len(first_batch_out)), [n[0] for n in first_batch_out.detach().numpy()], label="Predicted")
        plt.scatter(range(len(first_batch_y)), [n.detach() for n in first_batch_y], label="Correct")
        plt.title("Predicted and correct (first 100 subjects) - Epoch Final")
        plt.xlabel("Subject")
        plt.ylabel("Value")
        plt.legend(loc='upper right')
        plt.savefig(f"images/cnn-1-pred-vs-correct-epoch-final-{test_case}.png")
        plt.close()

        plt.plot(losses, label="Loss")
        plt.title("Loss per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')

        plt.savefig(f"images/cnn-1-loss-{test_case}.png")
        plt.close()

        plt.plot(corrs, label="Correlation")
        plt.title("Correlation between prediction and correct per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.legend(loc='upper right')
        plt.savefig(f"images/cnn-1-correlation-{test_case}.png")
        plt.close()


def load_data(filename):
    with open(filename) as fp:
        line = fp.readline()
        column_count = len(line.split(","))
    value_columns = [str((i + 1)) for i in range(column_count - 1)]
    labels = ["value"] + value_columns
    df = pd.read_csv(filename, names=labels)
    # torch.tensor(n, dtype=torch.float)
    x = [n for n in df[value_columns].values.tolist()]
    y = torch.tensor([n for n in df["value"].values], dtype=torch.float)
    torch_dataset = TensorDataset(torch.tensor(x, dtype=torch.float), y)

    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=100,
        shuffle=True, num_workers=2, )
    # data = torch.tensor(x)
    # data.y = y
    # mask = [1 if random() > 0.8 else 0 for _ in range(len(y))]
    # data.train_mask = torch.tensor(mask, dtype=torch.bool)
    # data.test_mask = torch.tensor([0 if n else 1 for n in mask], dtype=torch.bool)
    return loader, column_count-1


if __name__ == "__main__":
    test_case = "approx-patwa"
    loader, size = load_data("SNP.csv")  #

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net(dataset).to(device)
    # print((dataset[0]))

    model = Sequential(
        Conv1d(in_channels=1, out_channels=1, kernel_size=30),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=2),
        # Linear(size, 100),
        # Linear(100, 100),
        # Conv1d(in_channels=100, out_channels=100, kernel_size=1, bias=False),
        Flatten(),
        Linear(4847, 1),
    )
    # data = dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=5e-4)

    train(loader, plot=True)
