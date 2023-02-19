import torch


def parse_desc(desc: str):
    assert '->' in desc
    in_desc, out_desc = desc.split('->')
    in_desc = [x.strip() for x in in_desc.split(',') if len(x.strip()) > 0]
    out_desc = [x.strip() for x in out_desc.split(',') if len(x.strip()) > 0]
    return in_desc, out_desc


# from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/sequential.html#Sequential
class Sequential(torch.nn.Module):
    r"""An extension of the :class:`torch.nn.Sequential` container in order to
    define a sequential GNN model.
    Since GNN operators take in multiple input arguments,
    :class:`torch_geometric.nn.Sequential` expects both global input
    arguments, and function header definitions of individual operators.
    If omitted, an intermediate module will operate on the *output* of its
    preceding module:

    .. code-block:: python

        from torch.nn import Linear, ReLU
        from torch_geometric.nn import Sequential, GCNConv

        model = Sequential('x, edge_index', [
            (GCNConv(in_channels, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, out_channels),
        ])

    where ``'x, edge_index'`` defines the input arguments of :obj:`model`,
    and ``'x, edge_index -> x'`` defines the function header, *i.e.* input
    arguments *and* return types, of :class:`~torch_geometric.nn.conv.GCNConv`.

    In particular, this also allows to create more sophisticated models,
    such as utilizing :class:`~torch_geometric.nn.models.JumpingKnowledge`:

    .. code-block:: python

        from torch.nn import Linear, ReLU, Dropout
        from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
        from torch_geometric.nn import global_mean_pool

        model = Sequential('x, edge_index, batch', [
            (Dropout(p=0.5), 'x -> x'),
            (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x1, edge_index -> x2'),
            ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            (global_mean_pool, 'x, batch -> x'),
            Linear(2 * 64, dataset.num_classes),
        ])

    Args:
        args (str): Global input arguments of the model.
        modules ([(str, Callable) or Callable]): A list of modules (with
            optional function header definitions).
    """
    def __init__(self, args: str, modules):
        super(Sequential, self).__init__()

        self.args = [x.strip() for x in args.split(',') if len(x.strip()) > 0]

        assert len(modules) > 0
        assert isinstance(modules[0], (tuple, list))

        self.nns, self.fns, self.descs = torch.nn.ModuleList(), [], []
        for i, module in enumerate(modules):
            if isinstance(module, (tuple, list)) and len(module) >= 2:
                assert len(module) == 2
                module, desc = module
                in_desc, out_desc = parse_desc(desc)
            elif isinstance(module, (tuple, list)):
                module = module[0]
                in_desc = out_desc = self.descs[i - 1][1]
            else:
                in_desc = out_desc = self.descs[i - 1][1]
            if isinstance(module, torch.nn.Module):
                self.nns.append(module)
                self.fns.append(None)
            else:
                self.nns.append(None)
                self.fns.append(module)
            self.descs.append((in_desc, out_desc))

    def reset_parameters(self):
        for nn in self.nns:
            if hasattr(nn, 'reset_parameters'):
                nn.reset_parameters()

    # We can't really type check due to the dynamic behaviour of `Sequential`:
    def forward(self, *args):
        """"""
        if len(args) == 1 and len(self.args) > 1:
            args = args[0]
        specified_args = self.args[:]
        missing_args = []
        while len(args) < len(specified_args) and specified_args[-1][-1] == "?":
            if specified_args[-1][-1] == "?":
                missing_args.append(specified_args[-1][:-1])
                specified_args = specified_args[:-1]
        specified_args = [arg.strip("?") for arg in specified_args]

        assert len(args) == len(specified_args)
        state = {key: arg for key, arg in zip(specified_args, args)}
        state["self"] = self
        for nn, fn, (in_desc, out_desc) in zip(self.nns, self.fns, self.descs):
            fn = nn if fn is None else fn

            out = fn(*[state[key] for key in in_desc if key not in missing_args])
            out = (out, ) if not isinstance(out, tuple) else out
            assert len(out) == len(out_desc)
            state.update({key: item for key, item in zip(out_desc, out)})

        return out[0] if len(out) == 1 else out

    def __repr__(self):
        fns = [
            nn if fn is None else fn.__name__
            for nn, fn in zip(self.nns, self.fns)
        ]
        return (self.__class__.__name__ + '(\n{}\n)').format('\n'.join(
            [f'  ({i}): {fn}' for i, fn in enumerate(fns)]))

    def l1_regularize(self, loss, l1_weight=0.1):
        l1_parameters = []
        for parameter in self.parameters():
            l1_parameters.append(parameter.view(-1))

        l1 = l1_weight * torch.abs(torch.cat(l1_parameters)).sum()

        return loss + l1

    def replace(self, index, module):
        if isinstance(module, torch.nn.Module):
            self.nns[index] = module
            self.fns[index] = None
        else:
            self.nns[index] = None
            self.fns[index] = module

    def get(self, index):
        if self.nns[index] is not None:
            return self.nns[index]
        return self.fns[index]
