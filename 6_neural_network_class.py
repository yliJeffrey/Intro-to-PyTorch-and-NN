# Part 6 
# Building Neutal Network Class

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set a random seed
torch.manual_seed(42)

## create the NN_Regression class
class NN_Regression(nn.Module):
    def __init__(self):
        super(NN_Regression, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1)

        # initialize activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x


## create an instance of NN_Regression
model = NN_Regression()

## create an input tensor
apartments_df = pd.read_csv('streeteasy.csv')
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values
X = torch.tensor(apartments_numpy, dtype=torch.float32)

## feedforward to predict rent
predicted_rents = model(X)

## show output
print(predicted_rents[:5])
print()



# set a random seed
torch.manual_seed(42)

## create the OneHidden class
class OneHidden(nn.Module):
    def __init__(self, numHiddenNodes):
        super(OneHidden, self).__init__()
        # initialize layers
        self.layer1 = nn.Linear(2, numHiddenNodes)
        self.layer2 = nn.Linear(numHiddenNodes, 1)

        # initialize activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


## create an instance of NN_Regression
model = OneHidden(10)

## create an input tensor
X = torch.tensor([3, 4.5], dtype=torch.float32)

## feedforward to predict rent
predictions = model(X)

## show output
print(predictions)