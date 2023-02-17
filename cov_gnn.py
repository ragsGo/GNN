

import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sparse

# From https://github.com/shuaiOKshuai/Tail-GNN/blob/main/layers/discriminator.py
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.d = nn.Linear(in_features, in_features, bias=True)
        self.wd = nn.Linear(in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    '''
    def weight_init(self, m):
        if isinstance(m, Parameter):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:                
                m.bias.data.uniform_(-stdv, stdv)
    '''

    def forward(self, ft):

        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)

        return self.sigmoid(prob)


class Relationv2(nn.Module):
    def __init__(self, in_features, out_features, ablation=0):
        super(Relationv2, self).__init__()

        self.gamma1_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma1_2 = nn.Linear(out_features, in_features, bias=False)

        self.gamma2_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma2_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta1_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta1_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta2_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta2_2 = nn.Linear(out_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.ablation = ablation
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()

    def weight_init(self, m):
        return

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:
            gamma1 = self.gamma1_2(self.gamma1_1(ft))
            gamma2 = self.gamma2_2(self.gamma2_1(neighbor))
            gamma = self.lrelu(gamma1 + gamma2) + 1.0

            beta1 = self.beta1_2(self.beta1_1(ft))
            beta2 = self.beta2_2(self.beta2_1(neighbor))
            beta = self.lrelu(beta1 + beta2)

            self.r_v = gamma * self.r + beta
            self.m = ft + self.r_v - neighbor

        return F.normalize(self.m)

class Generator(nn.Module):
    def __init__(self, in_features, std, ablation):
        super(Generator, self).__init__()

        self.g = nn.Linear(in_features, in_features, bias=True)
        self.std = std
        self.ablation = ablation

    def forward(self, ft):
        # h_s = ft
        if self.training:
            # if self.ablation == 2:
            mean = torch.zeros(ft.shape, device='cuda')
            ft = torch.normal(mean, 1.)
            # else:
            #    ft = torch.normal(ft, self.std)
        h_s = F.elu(self.g(ft))

        return h_s


class TransSAGE(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma, device, ver, ablation=0, nheads=3, dropout=0.5, concat=True):
        super(TransSAGE, self).__init__()

        self.device = device
        self.ablation = ablation

        self.r = Relationv2(nfeat, nhid, ablation)
        self.g = Generator(nfeat, g_sigma, ablation)
        self.weight = nn.Linear(nfeat, nhid, bias=False)

    def forward(self, x, adj, head):

        mean = F.normalize(adj, p=1, dim=1)
        neighbor = torch.mm(mean, x)
        output = self.r(x, neighbor)

        if head or self.ablation == 2:
            ft_input = self.weight(x)
            ft_neighbor = self.weight(neighbor)
            h_k = torch.cat([ft_input, ft_neighbor], dim=1)

        else:
            if self.ablation == 1:
                h_s = self.g(output)
            else:
                h_s = output

            norm = torch.sum(adj, 1, keepdim=True) + 1
            neighbor = neighbor + h_s / norm
            ft_input = self.weight(x)
            ft_neighbor = self.weight(neighbor)
            h_k = torch.cat([ft_input, ft_neighbor], dim=1)

        return h_k, output


# latent relation GCN
class TailGNN(nn.Module):
    def __init__(self, nfeat, nclass, params, device, ver=1):
        super(TailGNN, self).__init__()

        defaults = {
            "hidden": 32,
            "dropout": 0.5,
            "g_sigma": 1,
            "ablation": 0,
        }
        params = {**defaults, **params}
        self.nhid = params["hidden"]
        self.dropout = params["dropout"]
        self.rel1 = TransSAGE(nfeat, self.nhid, g_sigma=params["g_sigma"], device=device,
                              ver=ver, ablation=params["ablation"])
        self.rel2 = TransSAGE(self.nhid * 2, nclass, g_sigma=params["g_sigma"], device=device,
                              ver=ver, ablation=params["ablation"])
        self.fc = nn.Linear(nclass * 2, nclass, bias=True)

    def forward(self, x, adj, head):
        x1, out1 = self.rel1(x, adj, head)
        x1 = F.elu(x1)
        x1 = F.normalize(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2, out2 = self.rel2(x1, adj, head)
        x2 = F.elu(x2)
        x2 = F.normalize(x2)
        x2 = self.fc(x2)

        return x2, F.relu(x2), [out1, out2]

    def embed(self, x, adj):
        x1, m1 = self.rel1(x, adj, False)
        x1 = F.elu(x1)
        x2, m2 = self.rel2(x1, adj, False)
        return F.relu(x2)


# based on https://theaisummer.com/graph-convolutional-networks/
def find_eigmax(l):
    with torch.no_grad():
        e1, _ = torch.eig(l, eigenvectors=False)
        return torch.max(e1[:, 0]).item()


def chebyshev_Lapl(x, lapl, thetas, order, max_neighbours):
    list_powers = []
    nodes = lapl.shape[0]

    t0 = x.float()

    eigmax = find_eigmax(lapl)
    l_rescaled = lapl.fill_diagonal_(0) #- torch.eye(nodes) #(2 * lapl / eigmax) - torch.eye(nodes)
    new = torch.zeros(nodes, nodes)
    for i in range(nodes):
        if torch.sum(l_rescaled[i, :]) > 0:
            idx = torch.multinomial(l_rescaled[i,:], max_neighbours)
            new[i, idx] = l_rescaled[i, idx]
    l_rescaled = new #torch.zeros(nodes, nodes).scatter_(0, torch.multinomial(l_rescaled, max_neighbours).t(), l_rescaled)

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


class CovGnn(nn.Module):
    """
    A simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,  power_order, max_neighbours=2, thetas=None, bias=True):
        super().__init__()
        if thetas is None:
            thetas = np.array(range(power_order, 0, -1))
            thetas = thetas/np.sum(thetas)
        self.thetas = nn.Parameter(torch.tensor(thetas))
        self.order = power_order
        self.max_neighbours = max_neighbours
        self.linear = nn.Linear(in_features*power_order, out_features, bias=bias)

    def forward(self, x, edge_index):
        """
        A: adjecency matrix
        X: graph signal
        """
        l = create_graph_lapl_norm(edge_index.type(torch.float))
        if self.order > 1:
            x = chebyshev_Lapl(x, l, self.thetas, self.order, self.max_neighbours)
        x = self.linear(x)
        return torch.bmm(l.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
