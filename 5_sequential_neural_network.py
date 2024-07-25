# Part 5
# Build a Sequential Neural Network
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn

# set a random seed - initial weights and biases are the same every time we run the same code
torch.manual_seed(42)

# create a neural network
model = nn.Sequential(nn.Linear(3, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1)
                     )
# show model details
print(model)
print()


# add a second hidden layer with four nodes and nn.Sigmoid() activation function
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(3, 8),
                      nn.ReLU(),
                      nn.Linear(8, 4),
                      nn.Sigmoid(),
                      nn.Linear(4, 1)
                     )
print(model)
print()

# Dataset Import
# load pandas DataFrame
apartments_df = pd.read_csv('streeteasy.csv')

# create a numpy array of the numeric columns
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values

# convert to an input tensor
X = torch.tensor(apartments_numpy, dtype=torch.float32)

# preview the first five apartments
print(X[:5])
print()


# model feedforward
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(3, 16),
                      nn.ReLU(),
                      nn.Linear(16, 8),
                      nn.ReLU(),
                      nn.Linear(8, 4),
                      nn.ReLU(),
                      nn.Linear(4, 1)
                     )
predicted_rent = model(X)
print(predicted_rent[:5])
