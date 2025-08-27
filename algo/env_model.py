import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class baseNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(baseNet, self).__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return F.softmax(x, dim=1)

class trainer():
    def __init__(self, input_dim, output_dim,):
        self.model = baseNet(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, x_train, y_train, num_epochs=1000, batch_size=32):
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).long()
        train_size = int(0.8 * len(x_train))
        test_size = len(x_train) - train_size
        train_x, test_x = torch.split(x_train, [train_size, test_size])
        train_y, test_y = torch.split(y_train, [train_size, test_size])

        loss_list = []

        for epoch in range(num_epochs):
            perm = torch.randperm(train_size)
            train_x = train_x[perm]
            train_y = train_y[perm]
            batches = train_size // batch_size
            for i in range(batches):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss_list.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        with torch.no_grad():
            output = self.model(test_x)
            loss = self.criterion(output, test_y)
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


        plt.plot(loss_list)
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    