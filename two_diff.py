
import torch
from torch_geometric.nn import GCNConv

from multi_gnn import GCNAISUMMER


class TwoDiffGNN(torch.nn.Module):
    def __init__(self, gnn1=None, gnn2=None, input_size=None, output_size=None, radius=None):
        super().__init__()
        assert (gnn1 is None and gnn2 is None and input_size is not None and output_size is not None ) or (input_size is None and output_size is None and radius is None)
        if radius is None:
            radius = 2
        self.gnn1 = gnn1 if gnn1 else GCNConv(input_size, output_size//2)
        self.gnn2 = gnn2 if gnn2 else GCNAISUMMER(input_size, output_size//2, power_order=radius)

    def reset_parameters(self):
        if hasattr(self.gnn, 'reset_parameters'):
            self.gnn.reset_parameters()

    def forward(self, *args):
        if len(args) == 1 and len(self.args) > 1:
            args = args[0]
        x = args[0] if isinstance(args[0], tuple) else (args[0], args[0])
        edges = args[1]
        assert len(edges) == 2
        out0 = self.gnn1(x[0], edges[0])
        out1 = self.gnn2(x[1], edges[1])
        return torch.cat((out0, out1), dim=1)
