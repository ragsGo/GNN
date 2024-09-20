
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from mlxtend.plotting import heatmap
import os
import matplotlib.ticker as ticker
from gnn.loaders.util import split_dataset_graph
import pathlib
from gnn.loaders.load import load_data
import pandas as pd
import networkx as nx

m = nn.Hardshrink(lambd=0.23)
input = torch.randn(2)
print(input)
print(type(input))
output = m(input)
print(output)
