import sys
sys.path.append('../..')
import torch
import numpy as np
from rw_train import Net
from matplotlib import pyplot as plt
from envs.random_walk.random_walk import RandomWalkEnv

def compute_traj_prob(trajs, model):
    log_prob_list = []
    for traj in trajs:
        with torch.no_grad():
            states = torch.tensor([transition[0] for transition in traj], dtype=torch.float32).view(-1, 1)
            actions = torch.tensor([transition[1] for transition in traj], dtype=torch.long)

            outputs = model(states)
            log_probs = torch.log(outputs)

            traj_log_probs = log_probs[range(len(actions)), actions]
            log_prob = traj_log_probs.sum().item()

        log_prob_list.append(log_prob)
    avg_log_prob = np.mean(log_prob_list)
    # print("Average log probability: ", avg_log_prob)
    return avg_log_prob


def test_likelihood(p=0.5, episodes=100):
    testNet = Net()
    testNet.load_state_dict(torch.load(f'transitionNet_{p}.pt'))
    testNet.eval()  # Set the network to evaluation mode
    
    env = RandomWalkEnv()

    trajectories = []
    for _ in range(episodes):
        transitions = []
        state = env.reset()
        done = False
        while not done:
            state, delta_s, done = env.step(p)
            label = 0 if delta_s == -1 else 1
            if done:
                label = 2
            transitions.append([state, label])
    trajectories.append(transitions)
    log_likelihood = compute_traj_prob(trajectories, testNet)
    prob = np.exp(log_likelihood)
    print("test probability: ", p)
    print("Average log probability: ", log_likelihood)
    print("Probability: ", prob)
    return log_likelihood, prob

def test_ratio(default_p=0.5, test_p=0.5, episodes=100):
    testNet0 = Net()
    testNet1 = Net()
    testNet0.load_state_dict(torch.load(f'transitionNet_{default_p}.pt'))
    testNet1.load_state_dict(torch.load(f'transitionNet_{test_p}.pt'))
    testNet0.eval()  # Set the network to evaluation mode
    
    env = RandomWalkEnv()

    trajectories = []
    for _ in range(episodes):
        transitions = []
        state = env.reset()
        done = False
        while not done:
            state, delta_s, done = env.step(test_p)
            label = 0 if delta_s == -1 else 1
            if done:
                label = 2
            transitions.append([state, label])
    trajectories.append(transitions)
    log_likelihood0 = compute_traj_prob(trajectories, testNet0)
    log_likelihood1 = compute_traj_prob(trajectories, testNet1)

    ratio = (log_likelihood0 - log_likelihood1)
    print("default probability: ", default_p)
    print("test probability: ", test_p)
    print("Likelihood ratio: ", ratio)
    return ratio


if __name__ == '__main__':
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # test_likelihood()
    prob_list = []
    for p in p_values:
        _, prob = test_likelihood(p=p)
        prob_list.append(prob)
    plt.plot(p_values, prob_list)
    plt.xlabel('p value')
    plt.ylabel('likelihood')
    plt.savefig('likelihood.png')

    plt.close()

    # test_ratio()
    ratio_list = []
    for p in p_values:
        ratio = test_ratio(p)
        ratio_list.append(ratio)
    plt.plot(p_values, ratio_list)
    plt.xlabel('p value')
    plt.ylabel('likelihood ratio')
    plt.savefig('likelihood_ratio.png')




