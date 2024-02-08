import numpy as np
import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def reverse(model, optim, input, output, inp_size, edges, loss_func=torch.nn.MSELoss(), num_steps=300, select_size=(-1, -1), discrete=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model.to(device)
    # print(model.get_device())
    model.requires_grad_(False)
    x = torch.rand(inp_size, requires_grad=True, device="cuda")

    optim.add_param_group({"params": x})

    for epoch in range(num_steps):
        optim.zero_grad()
        # x = torch.nn.Parameter(torch.rand(inp_size), requires_grad=True).cuda()
        loss = loss_func(model((x, *edges)), output.cuda())
        loss.backward()
        optim.step()
        print('Reverse epoch: {:03d}, Loss: {:.10f}'.format(epoch, loss))

    if isinstance(select_size, int):
        select_size = (select_size, select_size)

    y_size = select_size[1]
    if y_size < 0:
        y_size = len(output)
    x_size = select_size[0]
    if x_size < 0:
        x_size = inp_size[0]

    selections = list(range(0, inp_size[0], int(inp_size[0]/y_size)))
    values = x.detach().cpu().numpy()[selections, :x_size]
    # print (values)




    ylabels = output.cpu().squeeze().numpy()[selections]
    # print(max(ylabels))
    # max_y = int(ylabels.argmax())

    # vals = values[:,max_y]
    df = pd.DataFrame(values) #convert to a dataframe
    df['individuals'] = ylabels

    #df.corrwith(df['individuals']).plot(kind='barh')
    # sns.heatmap(df.corr().loc[:,['individuals']])
    corr = df.corr().loc[:,['individuals']]
    # print(type(input))
    # # plt.show()
    df_trns = pd.DataFrame(input.detach().cpu().numpy())
    df_trns.insert(0,'individuals',corr)
    # df_trns['individuals'] = corr

    df_trns.to_csv("Values.csv",index=False,header=None) #save to file

    if discrete:
      values = np.around(values)

    # key_vals = list(zip(values, ylabels))
    key_vals = list(zip(input.detach().cpu().numpy(), corr))

    values, ylabels = zip(*sorted(key_vals, key=lambda x: x[1]))
    # print(values, type(values))
    # vals = np.asarray(values)[:,max_y]
    # df = pd.DataFrame(vals) #convert to a dataframe
    # df.to_csv("Values.csv",index=False, mode='a', header=False)
    # plt.figure(figsize=(12, 50))
    plt.imshow(values)
    # plt.plot(vals)
    plt.xlabel(r'Loc', fontsize=12)
    plt.ylabel(r'Value', fontsize=12)
    # plt.yticks(np.arange(0, len(selections)), ylabels)
    plt.colorbar()
    # plt.show()

    return x
