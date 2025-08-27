import os
import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from torch import nn
import torch
import torch.nn.functional as F
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from solverScript import *
import sys
sys.path.append('../../')
from envs.bandit.bandit import BernoulliBandit

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
def collect_trajectories(solver, episodes=10):
    actions = []
    rewards = []
    for _ in range(episodes):
        solver.run(5000)
        actions.append(np.array(solver.actions).reshape(-1, 1))
        rewards.append(np.array(solver.rewards).reshape(-1, 1))
        solver.reset()
    # actions = np.array(actions).reshape(-1, 1)
    # rewards = np.array(rewards).reshape(-1, 1)
    
    return actions, actions, rewards


def bootstrap(observations, actions, delta_observations, size):
    indices = np.random.choice(len(observations), size, replace=True)
    
    return observations[indices], actions[indices], delta_observations[indices]


class TransitionModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim_1=32, hidden_dim_2=64):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim_2, 2)
        self.log_std = nn.Linear(hidden_dim_2, 2)
    
    def forward(self, obs, action):
        # x = torch.cat([obs, action], dim=-1)
        x = self.fc(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        # x = F.sigmoid(x)  # apply sigmoid to the output
        return mean, std
    
    def sample(self, obs, action):
        mean, std = self.forward(obs, action)
        dist = torch.distributions.Normal(mean, std)
        pred = dist.sample()
        return pred


def train_transition_model(model, observations, actions, delta_observations, epochs=100, batch_size=32, save_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(observations, dtype=torch.float32),
                                             torch.tensor(actions, dtype=torch.float32),
                                             torch.tensor(delta_observations, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    loss_list = []
    for epoch in range(epochs):
        for obs_batch, action_batch, delta_obs_batch in loader:
            mean, std = model(obs_batch, action_batch)
            dist = torch.distributions.Normal(mean, std)
            loss = -dist.log_prob(delta_obs_batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print(f'Epoch {epoch}: Loss = {loss.item()}')
    print()
    if save_dir is None:
        save_dir = 'transition_model.pth'
    torch.save(model.state_dict(), save_dir)
    # plt.plot(loss_list)
    # plt.savefig('transition_model_loss_8.png')


def train_ensemble_models(n_models, observations, actions, delta_observations, obs_dim, action_dim=1, epochs=100, batch_size=32, lr=0.001):
    models = []
    for i in range(n_models):
        print(f'Training model {i+1}/{n_models}')
        obs, action, delta_obs = bootstrap(observations, actions, delta_observations, size=500)
        model = TransitionModel(obs_dim, action_dim)
        train_transition_model(model, obs, action, delta_obs, epochs, batch_size, save_dir=f'transition_model_{i}.pth')
        models.append(model)
    return models


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    
    bandit = BernoulliBandit(p=0.1)
    solver = ThompsonSampling(bandit)
    solver.policy_fixed = True
    obs, actions, delta_obs = collect_trajectories(solver, cfg.detect.collect_trajs)
    obs = np.array(obs).reshape(-1, 1)
    actions = np.array(actions).reshape(-1, 1)
    delta_obs = np.array(delta_obs).reshape(-1, 1)
    # use model ensemble
    # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]

    # train transition model
    # transition_model = TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1)
    # transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
    #                                                                 f'transition_model_0_16.pth')))
    # save_dir = os.path.join(cfg.train.save_dir, 'acrobot_models', f'transition_model_0_16_new.pth')
    # train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)

    # # train ensemble models
    # train_ensemble_models(cfg.train.model_num, obs, actions, delta_obs, obs_dim=1, epochs=100, batch_size=32, lr=0.001)


    # for i in range(1, len(contexts)):
    # # i = 8
    #     env.context_id = i
    #     # render = lambda: plt.imshow(env.render())
    #     env.reset()
    #     print(f"Currently using {cfg.detect.param_name}: ", env.context[cfg.detect.param_name])
    #     policy = PPO.load(cfg.train.load_dir)
    #     # trajectories = collect_trajectories(env, policy, episodes=cfg.train.collect_episodes)
    #     # obs, actions, delta_obs = trajs_to_trans(trajectories)
    #     obs, actions, delta_obs = collect_transitions(env, policy, num_steps=cfg.train.collect_steps)
    #     # use model ensemble
    #     # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]
    #     transition_model = TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1)
    #     transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
    #                                                                     f'transition_model_0_16.pth')))
    #     save_dir = os.path.join(cfg.train.save_dir, 'acrobot_models', cfg.detect.param_name, f'transition_model_{i}.pth')
    #     train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)


if __name__ == '__main__':
    main()