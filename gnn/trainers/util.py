import torch


def l1_regularize(model, loss, l1_lambda):
    if hasattr(model, "l1_regularize"):
        return model.l1_regularize(loss, l1_lambda)
    l1_parameters = []
    for parameter in model.parameters():
        l1_parameters.append(parameter.view(-1))

    l1 = l1_lambda * torch.abs(torch.cat(l1_parameters)).sum().float()

    return loss + l1


def get_args_tuple(data):
    if hasattr(data, "edge_index") and data.edge_index is not None:
        if hasattr(data, "edge_weight") and data.edge_weight is not None:  # ** to prevent accessing float on None
            args_tuple = data.x.float(), data.edge_index, data.edge_weight.float()
        else:
            args_tuple = data.x, data.edge_index
    else:
        args_tuple = data.x
    return args_tuple
