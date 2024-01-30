import numpy as np
import torch


import matplotlib.pyplot as plt


def reverse(model, optim, output, inp_size, edges, loss_func=torch.nn.MSELoss(), num_steps=250, select_size=(100, 100)):
    model.requires_grad_(False)
    x = torch.nn.Parameter(torch.rand(inp_size), requires_grad=True).float()
    for _ in range(num_steps):
        loss = loss_func(model((x.float(), *edges)), output.float())
        loss.float().backward()
        optim.step()

    if isinstance(select_size, int):
        select_size = (select_size, select_size)

    selections = list(range(0, inp_size[0], int(inp_size[0]/select_size[1])))
    values = x.detach().numpy()[selections, :select_size[0]]
    ylabels = output.detach().squeeze().numpy()

    plt.figure(figsize=(12, 50))
    plt.imshow(values)
    plt.xlabel(r'Loc', fontsize=12)
    plt.ylabel(r'Value', fontsize=12)
    plt.yticks(np.arange(0, len(selections)), ylabels[selections])
    plt.colorbar()
    plt.show()

    return x
