import sys
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

from envs.random_walk.random_walk import RandomWalkEnv

def save_traj_csv(trajectories, p):
    df = pd.DataFrame()
    for i, trajectory in enumerate(trajectories):
        df = df.append(pd.Series(trajectory, name=f'trajectory_{i}'))
    # df = df.T
    df.to_csv(f'trajectories_{p}.csv', header=False, index=False, encoding='utf_8_sig')
    print('Trajectories saved')

# Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 3)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # prob for two actions
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#         self.fc1 = nn.Linear(1, 10)  # Input layer to hidden layer
#         self.fc2 = nn.Linear(10, 10)  # Hidden layer to hidden layer
#         self.fc3 = nn.Linear(10, 3)  # Hidden layer to output layer

#     def forward(self, x):
#         identity = x
#         out = torch.relu(self.fc1(x))
#         out = self.fc2(out)
#         out += identity  # Add skip (residual) connection
#         out = torch.relu(out)
#         out = torch.softmax(self.fc3(out), dim=1)
#         return out

def train(train_p=0.5, default_p=0.5, episodes=200, batch_size=32):
    # Initialize the neural network
    # transitionNet = Net()
    # # use default model to initialize the network
    # transitionNet.load_state_dict(torch.load(f'transitionNet_{default_p}.pt'))
    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(transitionNet.parameters(), lr=0.01)

    env = RandomWalkEnv()
    states = []
    labels = []

    trajectories = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        transitions = []
        while not done:
            state, delta_s, done = env.step(train_p)
            states.append(state)
            label = 0 if delta_s == -1 else 1
            if done:
                label = 2
            labels.append(label)
            transitions.append(state)
            transitions.append(label)
        trajectories.append(transitions)

    # save_traj_csv(trajectories, train_p)
    # df = pd.DataFrame({'state': states, 'label': labels})
    # df.to_csv(f'train_{train_p}.csv', index=False)

    X = torch.tensor(states, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(labels, dtype=torch.long).view(-1)

    X_test = torch.tensor(states, dtype=torch.float32).view(-1, 1)
    Y_test = torch.tensor(labels, dtype=torch.long).view(-1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a TensorDataset and DataLoader for training set
    train_dataset = TensorDataset(X_train, Y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a TensorDataset and DataLoader for validation set
    val_dataset = TensorDataset(X_val, Y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to save losses
    train_losses = []
    val_losses = []

    epochs = 200
    for epoch in range(epochs):
        for batch_X, batch_Y in train_dataloader:
            outputs = transitionNet(batch_X)
            loss = criterion(outputs, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save train loss
        train_losses.append(loss.item())

        # Validation
        if epoch % 10 == 0:
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_Y in val_dataloader:
                    outputs = transitionNet(batch_X)
                    loss = criterion(outputs, batch_Y)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Val Loss: {val_loss/len(val_dataloader)}')

    torch.save(transitionNet.state_dict(), f'transitionNet_{train_p}.pt')

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(range(0, len(train_losses), 10), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{train_p}.png')
    plt.close()

if __name__ == '__main__':
    # for p in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
    #     train(train_p=p)
    train(train_p=0.1)
    # train(train_p=0.9)