import numpy as np
import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def reverse(
    model,
    optim,
    output,
    inp_size,
    edges,
    loss_func=torch.nn.MSELoss(),
    num_steps=250,
    select_size=(100, 100),
    sort_labels=True,
    plot=False,
    save_name="save.png"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model.to(device)
    print(next(model.parameters()).is_cuda)
    model.requires_grad_(False)
    x = torch.rand(inp_size, requires_grad=True, device="cuda")

    print(x.get_device())
    print(output.cuda().get_device())

    optim.add_param_group({"params": x})

    for epoch in range(num_steps):
        optim.zero_grad()
        # x = torch.nn.Parameter(torch.rand(inp_size), requires_grad=True).cuda()
        loss = loss_func(model.cuda()((x.cuda(), *edges)), output.cuda())
        loss.backward()
        optim.step()
        print('Reverse epoch: {:03d}, Loss: {:.10f}'.format(epoch, loss))

    if isinstance(select_size, int):
        select_size = (select_size, select_size)

    range_size = int(inp_size[0] / select_size[1])
    selections = list(range(0, inp_size[0], range_size if range_size > 0 else 1))
    values = x.cpu().detach().numpy()[selections, : select_size[0]]

    if plot:
        y_labels = output.cpu().detach().squeeze().numpy()
        if sort_labels:
            val_labels = zip(values, y_labels)
            values, y_labels = zip(*sorted(list(val_labels), key=lambda k: k[1]))
        plt.figure(figsize=(12, 50))
        plt.imshow(values)
        plt.xlabel(r"Loc", fontsize=12)
        plt.ylabel(r"Value", fontsize=12)
        plt.yticks(np.arange(0, len(selections)), y_labels[selections])
        plt.colorbar()
        plt.savefig(save_name)

    return x
