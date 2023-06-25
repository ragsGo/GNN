# based on https://theaisummer.com/graph-convolutional-networks/

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sparse

def find_eigmax(l):
    with torch.no_grad():
        e1, _ = torch.eig(l, eigenvectors=False)
        return torch.max(e1[:, 0]).item()


def chebyshev_Lapl(x, lapl, thetas, order):
    list_powers = []
    nodes = lapl.shape[0]

    t0 = x.float()

    eigmax = find_eigmax(lapl)
    l_rescaled = (2 * lapl / eigmax) - torch.eye(nodes)

    y = t0 * thetas[0]
    list_powers.append(y)
    t1 = torch.matmul(l_rescaled, t0)
    list_powers.append(t1 * thetas[1])

    # Computation of: T_k = 2*L_rescaled*T_k-1  -  T_k-2
    for k in range(2, order):
        t2 = 2 * torch.matmul(l_rescaled, t1) - t0
        list_powers.append((t2 * thetas[k]))
        t0, t1 = t1, t2
    y_out = torch.stack(list_powers, dim=-1)
    # the powers may be summed or concatenated. i use concatenation here
    y_out = y_out.view(nodes, -1)  # -1 = order* features_of_signal
    return y_out


def device_as(x, y):
    device = y.device if hasattr(y, "device") else "cpu"
    return x.to(device)


# tensor operations now support batched inputs
def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    a += device_as(torch.eye(size), a)
    d_norm = calc_degree_matrix_norm(a)
    a = a.unsqueeze(0)
    d_norm = d_norm.unsqueeze(0)
    l_norm = torch.bmm(torch.bmm(d_norm, a), d_norm)
    return l_norm.squeeze(0)


class GCNAISUMMER(nn.Module):
    """
    A simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,  power_order, thetas=None, bias=True):
        super().__init__()
        if thetas is None:
            thetas = np.square(range(power_order, 0, -1))
            thetas = thetas/np.sum(thetas)
        self.thetas = nn.Parameter(torch.tensor(thetas))
        self.order = power_order
        self.linear = nn.Linear(in_features*power_order, out_features, bias=bias)

    def forward(self, x, edge_index):
        """
        A: adjecency matrix
        X: graph signal
        """
        l = create_graph_lapl_norm(edge_index.type(torch.float))
        if self.order > 1:
            x = chebyshev_Lapl(x, l, self.thetas, self.order)
        x = self.linear(x)
        return torch.bmm(l.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
