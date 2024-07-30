# Part 11
# Structured Training
# define functions for each part of the training process

import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


epochs = 1000

# import/load data
def load_data():
    apartments_df = pd.read_csv("streeteasy.csv")
    # 14 input features
    numerical_features = ['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs',
                          'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher',
                          'has_patio', 'has_gym']

    # create tensor of input features
    X = torch.tensor(apartments_df[numerical_features].values, dtype=torch.float)
    # create tensor of targets
    y = torch.tensor(apartments_df['rent'].values, dtype=torch.float).view(-1, 1)

    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.3, random_state=2)
    return X_train, X_test, y_train, y_test


# create model
def create_model():
    # set a random seed
    torch.manual_seed(42)

    # Define the model using nn.Sequential
    model = nn.Sequential(
        nn.Linear(14, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return model


def train(model, epochs, X_train, y_train):
    # MSE loss function + optimizer
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        predictions = model(X_train)
        MSE = loss(predictions, y_train)
        MSE.backward()
        optimizer.step()
        optimizer.zero_grad()

        # keep track of the loss during training
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {MSE.item()}')
        
    # save model
    torch.save(model, 'model.pth')


def evaluate(model, X_test, y_test):
    loaded_model = torch.load('model.pth')
    loaded_model.eval()
    with torch.no_grad():
        predictions = loaded_model(X_test)
    return predictions


# Visualization
def visualize(model, X_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, evaluate(model, X_test, y_test), label='Predictions', alpha=0.5, color='blue')

    plt.xlabel('Actual Values (y_test)')
    plt.ylabel('Predicted Values')

    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', linewidth=2, label='Actual Rent')
    plt.legend()
    plt.title('StreetEasy Dataset - Predictions vs Actual Values')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = create_model()
    train(model, epochs, X_train, y_train)
    visualize(model, X_test, y_test)

    predictions = evaluate(model, X_test, y_test)
    loss = nn.MSELoss()
    loaded_model = torch.load('model.pth')
    loaded_model.eval()
    with torch.no_grad():
        predictions = loaded_model(X_test)
        test_MSE = loss(predictions, y_test)
        # show output
        print('Test MSE is ' + str(test_MSE.item()))
        print('Test Root MSE is ' + str(test_MSE.item()**(1/2)))


if __name__ == '__main__':
    main()