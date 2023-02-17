import matplotlib.pyplot as plt
import numpy as np
import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
#from ax.utils.tutorials.cnn_utils import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from load_slp import load_data
from networks import create_network_linear,create_network_conv1D


def create_data(loader, params=None, test_case="default", plot=True):
    if params is None:
        params = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = load_data("SNP.csv", bits=["4354",  "931", "5321", "987" ,"1063", "5327" , "5350", "5322", "5333", "942" , "1014", "923", "1030", "1106", "979"])  #
    dataset = loader("SNP.csv", num_neighbours=3, smoothing="laplacian", mode="connectivity", use_validation=False)



    data = dataset[0].to(device)

    return data, dataset.num_features

def train( device ="cpu"):
    data, num_features = create_data(load_data)
    #poolsz = parameters.get("plsz", 2)
    #kr_sz = parameters.get("krsz", 30)
    model = create_network_conv1D(num_features, 1,)
    print('lalal,',model)

    #learning_rate = parameters.get("lr", 0.0006)
    #weight_decay= parameters.get("wd", 0.142904779816756e-05)#
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.000201033951607494779) #, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss()
    no_improvement_cap = 1000000
    no_improvement = 0
    test_loss = 100000000
    last_loss = test_loss
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        #print(data.x.unsqueeze(0).unsqueeze(0))
        out = model(data.x.unsqueeze(1))
        if len(data.test.x) > 0:
            out_test = model(data.test.x.unsqueeze(1))
            test_y = data.test.y
        else:
            out_test = model((data.x, data.edge_index))
            test_y = data.y

        loss = loss_func(out, data.y)
        l1_lambda = 0.1 #parameters.get("l1", 0.3)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda*l1_norm
        loss.backward()
        optimizer.step()
        test_loss = (float(loss_func(out_test, test_y)))
   
        print('epoch {}, loss {}, test_loss {}'.format(epoch, loss.item(), test_loss))
    #return test_loss

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train()
#best_parameters, values, experiment, model = optimize(
    #parameters=[
        #{"name": "lr", "type": "range", "bounds": [0.00015, 0.00045], "log_scale": True},
        #{"name": "l1", "type": "range", "bounds": [0.1, 1.0]},
        #{"name": "krsz", "type": "range", "bounds": [10, 50]},
        #{"name": "plsz", "type": "range", "bounds": [1, 5]},
        ##{"name": "wd", "type": "range", "bounds": [0.00001, 0.00002]},
        ##{"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        ##{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        ##{"name": "stepsize", "type": "range", "bounds": [20, 40]},        
    #],
  
    #evaluation_function=train_evaluate,
    #objective_name='mseloss',
    #minimize=True
#)

#print(best_parameters)
##means, covariances = values
#print(values)
##print(covariances)
