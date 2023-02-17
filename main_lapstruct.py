 from random import random

from torch.autograd import Variable
from torch.nn import ReLU, Linear, Sequential, Conv1d, Flatten, MaxPool1d
import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from lapstruct import lapstruct


def train(loader, test, plot=False):
    x_test, y_test = test
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

            b_x = Variable(batch_x.to(device))
            b_y = Variable(batch_y.to(device))

            # inp = data
            # print(inp)
            optimizer.zero_grad()
            out = model(b_x)
            loss = loss_func(out, b_y)
            loss.backward()
            optimizer.step()
            
            
            
            b_x = Variable(x_test.to(device))
            b_y = Variable(y_test.to(device))
            
            optimizer.zero_grad()
            out = model(b_x)
            loss = loss_func(out, b_y)
                        
            outs.append(out)
            individual_losses.append(float(loss))
            corr = np.corrcoef([n[0] for n in out.detach().numpy()], [n.detach() for n in b_y])
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
    G = pd.read_csv("SNP.csv", header=None)
    G_train = G.loc[:2326,:]
    G_test = G.loc[2326:,:]
    
    
    y_train = torch.tensor([n for n in G_train.iloc[:,0]], dtype=torch.float)
    G_train = G_train.iloc[ :, 1:]
    l = lapstruct(G_train.values)
    x_train = [n for n in l[:,-2:].tolist()]
    
    
    y_test = torch.tensor([n for n in G_test.iloc[:,0]], dtype=torch.float)
    G_test = G_test.iloc[ :, 1:]
    l_test = lapstruct(G_test.values)
    x_test = torch.tensor([n for n in l_test[:,-2:].tolist()], dtype=torch.float)
    
    
    torch_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float), y_train)

    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=50,
        shuffle=True, num_workers=2, )
    
    return loader, (x_test, y_test)


if __name__ == "__main__":
    test_case = "lapstruct"
    loader, test = load_data("SNP.csv")  #

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net(dataset).to(device)
    # print((dataset[0]))

    model = Sequential(
        Linear(2, 4),
        Linear(4, 4),
        Linear(4, 1),
    )
    # data = dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=5e-4)

    train(loader, test, plot=True)
