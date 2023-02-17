import numpy as np
import torch
from torch import nn
from torch.nn import ReLU, Conv1d, MaxPool1d, Flatten, Linear, Dropout, Conv2d, MaxPool2d, Identity
from torch.nn import functional as F
from sequential import Sequential

class Unsqeeze(nn.Module):
    def __init__(self, dim):
        super(Unsqeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

def create_network_conv1D(inp_size, out_size, conv_kernel_size=None, filters=None, as_double=False, **_):

    print('conv_kernel_size==', conv_kernel_size)
    print('filters==', filters)
    if filters is None:
        filters = [20,25]
    elif isinstance(filters, int):
        filters = [filters, filters+5]
    if conv_kernel_size is None:
        conv_kernel_size = [10, 15]
    elif isinstance(conv_kernel_size, int):
        conv_kernel_size = [conv_kernel_size, conv_kernel_size+5]
    out_conv = inp_size
    for i in range(2):
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size[i] - 1) - 1) / 1 + 1)
        out_conv = int((out_conv + 2 * 0 - 1 * (conv_kernel_size[i] - 1) - 1) / conv_kernel_size[i] + 1)

    out_conv *= filters[-1]
    # out_conv = int(out_conv/2)

    model = nn.Sequential(
        Unsqeeze(1),
        Conv1d(in_channels=1, out_channels=filters[0], kernel_size=conv_kernel_size[0], padding=0),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=conv_kernel_size[0], padding=0),
        Conv1d(in_channels=filters[0], out_channels=filters[1], kernel_size=conv_kernel_size[1], padding=0),
        ReLU(inplace=True),
        MaxPool1d(kernel_size=conv_kernel_size[-1], padding=0),
        Flatten(),
        Linear(out_conv, out_size),
        # Squeeze(1),
    )
    if as_double:
        model.double()
    return model