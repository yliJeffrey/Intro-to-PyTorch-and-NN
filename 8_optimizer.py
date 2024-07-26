# Part 8 
# Optimizer

import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn

# import optimizer
import torch.optim as optim

# set a random seed
torch.manual_seed(42)

# create neural network
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# import the data
apartments_df = pd.read_csv('streeteasy.csv')
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values
X = torch.tensor(apartments_numpy, dtype=torch.float32)
y = torch.tensor(apartments_df[['rent']].values, dtype=torch.float)

# forward pass
predictions = model(X)

# define the loss function and compute loss
loss = nn.MSELoss()                     
MSE = loss(predictions, y)          # output of the loss function includes the parameter grad_fn=<MseLosBackward0>, which is the function used to perform the backwards pass
print('Initial loss is ' + str(MSE))

# create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# backward pass to determine "downward" direction; backward() calculate the gradients of the loss function
MSE.backward()

# apply the optimizer to update weights and biases; step() - use the gradients to update teh weights and biases
optimizer.step()  


# feed the data through the updated model and compute the new loss
predictions = model(X)
MSE = loss(predictions, y)
print('After optimization, loss is ' + str(MSE))