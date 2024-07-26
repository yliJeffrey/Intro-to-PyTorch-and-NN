# Part 7
# The Loss function
# Mean Squared Error (MSE)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

difference1 = 750 - 1000
difference2 = 1000 - 900
MSE = (difference1**2 + difference2**2) / 2

print(MSE)


# define prediction and target tensors
predictions = torch.tensor([ -6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype=torch.float)
y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype=torch.float)

loss = nn.MSELoss()
MSE = loss(predictions, y)
print("MSE Loss: ", MSE)

# Root Mean Squared Error (RMSE)
RMSE = MSE**(1/2)
print(RMSE)