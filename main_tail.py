import os
import pickle
import time

import pandas as pd
from torch.nn import ReLU, Linear, MaxPool1d, Flatten, Conv1d
from torch import nn
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from load_rels import load_data as load_data_rels

import torch.nn.functional as F
import torch.optim as optim
import scipy as sp
import datetime, time
import collections, re
import os

from cov_gnn import Discriminator, TailGNN


def normalize_output(out_feat):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m, dim=1))
    return sum_m


def train_disc(disc, optimizer_D, embed_model, data):
    criterion = nn.BCELoss()
    disc.train()
    optimizer_D.zero_grad()
    embed_h, _, _ = embed_model(data.x, data.edge_index, True)
    embed_t, _, _ = embed_model(data.valid["x"], data.valid["edge_index"], False)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    # loss
    errorD = criterion(prob_h, data.y)
    errorG = criterion(prob_t, data.valid["y"])
    L_d = (errorD + errorG) / 2

    L_d.backward()
    optimizer_D.step()
    return L_d


def train_embed(disc, optimizer, embed_model, data):
    eta = 0.1
    mu = 0.001
    embed_model.train()
    optimizer.zero_grad()
    criterion = nn.BCELoss()

    embed_h, output_h, support_h = embed_model(data.x, data.edge_index, True)
    embed_t, output_t, support_t = embed_model(data.valid["x"], data.valid["edge_index"], False)

    # loss
    L_cls_h = F.mse_loss(output_h, data.y)
    L_cls_t = F.mse_loss(output_t, data.valid["y"])
    L_cls = (L_cls_h + L_cls_t) / 2

    # weight regularizer
    m_h = normalize_output(support_h)
    m_t = normalize_output(support_t)

    prob_h = disc(embed_h)
    prob_t = disc(embed_t)

    errorG = criterion(prob_t, data.valid["y"])
    L_d = errorG
    L_all = L_cls - (eta * L_d) + mu * m_h

    L_all.backward()
    optimizer.step()

    # validate:
    embed_model.eval()
    _, embed_val, _ = embed_model(data.test.x, data.test.edge_index, False)
    loss_val = F.mse_loss(embed_val, data.test.y)

    return (L_all, L_cls, L_d),  loss_val


def test(embed_model, data):
    embed_model.eval()
    _, embed_test, _ = embed_model(data.test.x, data.test.edge_index, False)
    loss_test = F.mse_loss(embed_test, data.test.y)

    log = "Test set results: " + \
          "loss={:.4f} ".format(loss_test.item())

    print(log)
    return


def main():
    dataset = load_data_rels("SNP.csv", use_validation=True)
    save_path = "model/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    data.x = data.x.float()
    data.test.x = data.test.x.float()
    data.valid["x"] = data.valid["x"].float()
    data.y = data.y.float()
    data.test.y = data.test.y.float()
    data.valid["y"] = data.valid["y"].float()
    data.edge_index = data.edge_index.float()
    data.test.edge_index = data.test.edge_index.float()
    data.valid["edge_index"] = torch.tensor(data.valid["edge_index"], dtype=torch.float32)

    # Model and optimizer
    embed_model = TailGNN(nfeat=dataset.num_features,
                          nclass=1,
                          params={},
                          device=device,
                          ver=2)

    optimizer = optim.Adam(embed_model.parameters(), lr=0.00025, weight_decay=0.00025)

    feat_disc = 1
    disc = Discriminator(feat_disc)
    optimizer_D = optim.Adam(disc.parameters(), lr=0.00025, weight_decay=0.00025)

    # if device == "cuda":
    #     embed_model = embed_model.cuda()
    #     disc = disc.cuda()

    best_loss = 10000.0
    loss_early_stop = 0.0
    cur_step = 0

    # Train model
    t_total = time.time()
    for epoch in range(5000):

        # L_d = train_disc(disc, optimizer_D, embed_model, data)
        L_d = train_disc(disc, optimizer_D, embed_model, data)

        Loss, loss_val,  = train_embed(disc, optimizer, embed_model, data)

        log = 'Epoch: {:d} '.format(epoch + 1) + \
              'loss_train: {:.4f} '.format(Loss[0].item()) + \
              'loss_val: {:.4f} '.format(loss_val)
        print(log)

        # save best model
        if loss_val <= best_loss:
            loss_early_stop = loss_val

            # torch.save(embed_model, os.path.join(save_path, 'tail_model.pt'))
            best_loss = np.min((loss_val, best_loss))
            print('Model saved!')

            cur_step = 0
        else:
            cur_step += 1
            if cur_step == 100:
                early_stop = 'Early Stopping at epoch {:d} '.format(epoch) + \
                             'loss {:.4f} '.format(loss_early_stop)
                print(early_stop)
                break

    print("Training Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    print('Test ...')
    embed_model = torch.load(os.path.join(save_path, 'tail_model.pt'))
    test(embed_model, data)


if __name__ == "__main__":
    main()
