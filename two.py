
import torch
from torch_geometric.nn import GCNConv


class TwoGNN(torch.nn.Module):
    def __init__(self, gnn=None, input_size=None, output_size=None):
        super().__init__()
        assert (gnn is None and input_size is not None and output_size is not None ) or (input_size is None and output_size is None)
        self.gnn = gnn if gnn else GCNConv(input_size, output_size//2)

    def reset_parameters(self):
        if hasattr(self.gnn, 'reset_parameters'):
            self.gnn.reset_parameters()

    def forward(self, *args):
        if len(args) == 1 and len(self.args) > 1:
            args = args[0]
        x = args[0] if isinstance(args[0], tuple) else (args[0], args[0])
        edges = args[1]
        assert len(edges) == 2
        out0 = self.gnn(x[0], edges[0])
        out1 = self.gnn(x[1], edges[1])
        return torch.cat((out0, out1), dim=1)
