
import os
import pathlib
import pickle
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy
from bayes_opt import BayesianOptimization
import torch
from bayes_opt import SequentialDomainReductionTransformer

import matplotlib.pyplot as plt
import numpy as np
from gnn.res_gate import create_data as create_res_data
from gnn.res_gate import create_graphs,collate, evaluate
from gnn.res_gate import GatedGCN
from gnn.res_gate import train as train_res

import warnings
warnings.filterwarnings("ignore")


def train(lr, hid_l, bt_sz,lmda):
    #
    # params = {
    #     'lr': lr,
    #     'hid_dim': int(hid_dim),
    #     'hid_l': int(hid_l),
    #
    # }
    EPOCHS = 4
    train_data, test_data, num_features = create_graphs("MiceBL.csv")

    train_loader = DataLoader(train_data, batch_size=int(bt_sz), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=int(bt_sz), shuffle=False, collate_fn=collate)


    model = GatedGCN(input_dim=num_features, hidden_dim=100, output_dim=1, L=int(hid_l,),activation=nn.LogSoftmax, lmd=lmda)


    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        train_loss = train_res(model, optimizer, train_loader, loss)
        test_loss = evaluate(model, optimizer, test_loader, loss)

    return -test_loss
        # test_loss = evaluate(model, optimizer, test_loader, loss)

        # print(f"Epoch {epoch}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")



    # test_loader = DataLoader(test_data, batch_size=50, shuffle=False, collate_fn=collate)


bounds = {
    'lr':(0.0000001,0.0001),
    'bt_sz': (50,500),
    'hid_l':(1,10),
    'lmda' : (0.001, 10)
    # 'act':[nn.LogSigmoid, nn.Sigmoid, nn.Softplus, nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax,]
  }

optimizer = BayesianOptimization(
    f=train,
    pbounds=bounds,
    random_state=1,
)

optimizer.maximize(init_points=10, n_iter=30)


