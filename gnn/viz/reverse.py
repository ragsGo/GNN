import numpy as np
import torch

import matplotlib.pyplot as plt


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
    model.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(inp_size, requires_grad=True, device=device)

    optim.add_param_group({"params": x})

    for epoch in range(num_steps):
        optim.zero_grad()
        # x = torch.nn.Parameter(torch.rand(inp_size), requires_grad=True)..to(device)
        loss = loss_func(model((x, *edges)), output.to(device))
        loss.backward()
        optim.step()
        print('Reverse epoch: {:03d}, Loss: {:.10f}'.format(epoch, loss))

    if isinstance(select_size, int):
        select_size = (select_size, select_size)

    range_size = int(inp_size[0] / select_size[1])
    selections = list(range(0, inp_size[0], range_size if range_size > 0 else 1))
    values = x.detach().numpy()[selections, : select_size[0]]

    if plot:
        y_labels = output.detach().squeeze().numpy()
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
