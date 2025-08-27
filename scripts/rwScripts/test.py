import sys
sys.path.append('../..')
import torch
import numpy as np
from rw_train import Net
from matplotlib import pyplot as plt
from envs.random_walk.random_walk import RandomWalkEnv


def test():
    testNet = Net()
    testNet.load_state_dict(torch.load(f'transitionNet_{0.1}.pt'))
    testNet.eval()  # Set the network to evaluation mode

    for i in range(5):
        i = torch.tensor([[i]]).float()
        prob = testNet(i)
        print(prob)

if __name__ == '__main__':
    test()