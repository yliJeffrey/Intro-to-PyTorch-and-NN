# Part 1
# Introduction to Tensors
import pandas as pd 
import torch
import numpy as np 

# create tensor
apartment_array = np.array([2000, 500, 7])
apartment_tensor = torch.tensor(apartment_array, dtype=torch.int)

# show output
print(apartment_tensor)
print()


# convert pandas DataFrame to Tensors
# import the dataset using pandas
apartment_df = pd.read_csv("streeteasy.csv")

# select the rent, size, and age columns
apartment_df = apartment_df[["rent", "size_sqft", "building_age_yrs"]]

apartment_tensor = torch.tensor(apartment_df.values, dtype=torch.float32)

# show output
print(apartment_tensor)
