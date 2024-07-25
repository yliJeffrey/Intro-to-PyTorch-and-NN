# Part 3
# Linear Regression with Perceptron

# Define the inputs
size_sqft = 1250.0
age = 15.0
bedrooms = 2.0
bias = 1

# The inputs flow through the edges, receiving weights
weighted_size = 3 * size_sqft
weighted_age = -2.3 * age
weighted_bedrooms = 100 * bedrooms
weighted_bias = 500 * bias

# The output node adds the weighted inputs
weighted_sum = weighted_size + weighted_age + weighted_bedrooms + weighted_bias

# Generate prediction
print("Predicted Rent:", weighted_sum)
