
import os
import pathlib
import pickle
import shutil
import time
import numpy as np

import scipy
from bayes_opt import BayesianOptimization
import torch
from bayes_opt import SequentialDomainReductionTransformer
from hyperopt import delete, save, load
from load_ensembles2 import load_data as load_data_ensemble2
from load import load_data
from main import create_data, l1_regularize, all_the_things,model_creator
from networks import create_network_conv_two_diff, create_network_conv1D, \
    create_network_two_diff, create_network_no_conv, create_network_no_conv, create_network_no_conv_dropout, \
    create_network_two_no_conv_dropout, create_network_two_no_conv_relu_dropout, create_network_no_conv_relu_dropout, \
    Ensemble
from load_two import load_data as load_data_two
from load_scaled import load_data as load_data_scaled
from load_batches import load_data as load_data_batches

from load_hot2 import load_data as load_data_hot2
import matplotlib.pyplot as plt
epochs =300
loss_func = torch.nn.MSELoss()
path = "model/1279-1-timed-350-train_ensemble2-load_data_ensembles2-WHEAT_combined.csv-1673522319.7359452.pt"
dataset = create_data(load_data, params={'filename': "csv-data/WHEAT1.csv"})
data, inp_size = dataset

aggr_model = create_network_two_no_conv_relu_dropout(inp_size,1,internal_size=100,dropout=0.4011713125675628,)
aggr_model.load_state_dict(torch.load(path))

optimizer = torch.optim.Adam(aggr_model.parameters(),lr=0.0020594745443455593)
if hasattr(data, "edge_weight") and data.edge_weight is not None:
    valid_tuple = data.x.float(), data.edge_index, data.edge_weight.float()
else:
    valid_tuple = data.x, data.edge_index

#aggr_model.train()

# aggr_loss = []
# aggr_model.eval()
#
# for epoch in range(epochs):
#     out = aggr_model(*valid_tuple)
#     #
#     loss = loss_func(out, data.y)
#     loss.backward()
#     optimizer.step()
#     print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss))
#     aggr_loss.append(float(loss))
#
# print(min(aggr_loss))

with torch.no_grad():
    aggr_model.eval()
    y_pred = aggr_model(*valid_tuple)
    val_loss = loss_func(y_pred, data.y)
