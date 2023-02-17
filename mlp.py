import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import matplotlib.pyplot as plt

# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
      
    def compute_l1_loss(self, w):
      return torch.abs(w).sum()
    
inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.001 
epochs = 100

model = linearRegression(inputDim, outputDim)
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    loss1 = loss
    l1_weight = 0.1
    l1_parameters = []
    ##print('model parms = ', model.parameters)
    for parameter in model.parameters():
        #print(parameter)
        l1_parameters.append(parameter.view(-1))
    print('len ==', len(l1_parameters))
    l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))
    
    # Add L1 loss component
    print ('before ==', loss1)
    print('actual ==',loss)
    print('l1 ===',l1)
    loss1 += l1
    print ('after ==', loss1)
    print('actual ==',loss)
    #print('l1 ==', l1)    
  
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    #print('epoch {}, loss {}, l1loss {}'.format(epoch, loss.item(), loss1.item()))
    
    
with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
