import os
import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from torch import nn
import torch
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from gymEnvGen import *

from gym_detect import *

def CARL_obs_transform(obs):
    if isinstance(obs, dict):
        return obs['obs']
    if isinstance(obs, tuple):
        return obs[0]['obs']
    if isinstance(obs, np.ndarray):
        return obs
    
def collect_trajectories(env, model, episodes=10):
    trajectories = []
    traj_ends = []
    for i in range(episodes):
        transitions = []
        obs = env.reset()
        obs = CARL_obs_transform(obs)
        done = False
        truncated = False
        steps = 0
        # begin token
        transition = [np.zeros_like(obs), np.array([0]), 0.0, obs]
        transitions.append(transition)
        while not (done or truncated):
            steps += 1
            action, _ = model.predict(obs, deterministic=False)
            new_obs, reward, done, truncated, info = env.step(action)
            new_obs = CARL_obs_transform(new_obs)
            delta_obs = new_obs - obs if not (done or truncated) else np.zeros_like(obs)
            transition = [obs, action.reshape(-1,), reward, delta_obs]
            transitions.append(transition)
            obs = new_obs
        trajectories.append(transitions)
        traj_ends.append(transition)
    return trajectories


def trajs_to_trans(trajectories):
    observations, actions, delta_obs_list = [], [], []
    for trj in trajectories:
        for transition in trj:
            observations.append(transition[0])
            actions.append(transition[1])
            delta_obs_list.append(transition[3])
    return np.array(observations), np.array(actions).reshape(-1, 1), np.array(delta_obs_list)

def collect_transitions(env, model, num_steps=1000):
    obs = env.reset()
    obs = CARL_obs_transform(obs)
    observations, actions, delta_obs_list = [], [], []
    # add start token
    observations.append(np.zeros_like(obs))
    actions.append(np.array(0))
    delta_obs_list.append(obs)
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=False)
        new_obs, _, done, _, _ = env.step(action)
        new_obs = CARL_obs_transform(new_obs)
        delta_obs = new_obs - obs
        observations.append(obs)
        actions.append(action)
        delta_obs_list.append(delta_obs)
        if done:
            done = False
            obs = env.reset()
            obs = CARL_obs_transform(obs)
            observations.append(np.zeros_like(obs))
            actions.append(np.array(0))
            delta_obs_list.append(obs)
        else:
            obs = new_obs
        
    return np.array(observations), np.array(actions).reshape(-1, 1), np.array(delta_obs_list)


def bootstrap(observations, actions, delta_observations, size):
    indices = np.random.choice(len(observations), size, replace=True)
    
    return observations[indices], actions[indices], delta_observations[indices]


class TransitionModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim_1=32, hidden_dim_2=64):
        super(TransitionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_1),
            nn.ReLU(),
            # nn.Linear(hidden_dim_1, hidden_dim_2),
            # nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim_1, obs_dim)
        self.log_std = nn.Linear(hidden_dim_1, obs_dim)
        self.output_type = 'gaussian'
    
    def forward(self, obs, action):
        # x = torch.cat([obs, action], dim=-1)
        x = self.fc(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, obs, action):
        mean, std = self.forward(obs, action)
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()


def train_transition_model(model, observations, actions, delta_observations, epochs=100, batch_size=32, save_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(observations, dtype=torch.float32),
                                             torch.tensor(actions, dtype=torch.float32),
                                             torch.tensor(delta_observations, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(dataset.observations)
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
        train_transition_model(model, obs, action, delta_obs, epochs, batch_size, save_dir=f'models/forward_models/transition_model_{i}.pth')
        models.append(model)
    return models


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    env_name = "Acrobot"
    param_name_list, policy_name = Parameter_name_list(env_name)
    
    # print(f"Currently using {cfg.detect.param_name}: ", env.context[cfg.detect.param_name])
    policy = PPO.load(os.path.join(cfg.train.load_dir, policy_name))
    # trajectories = collect_trajectories(env, policy, episodes=cfg.train.collect_episodes)
    # obs, actions, delta_obs = trajs_to_trans(trajectories)

    for param in param_name_list:
        env, contexts, _ = genEnv(env_name, param)
        # env.context_id = 0
        # # render = lambda: plt.imshow(env.render())
        # env.reset()

        # obs, actions, delta_obs = collect_transitions(env, policy, num_steps=cfg.train.collect_steps)
        # use model ensemble
        # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]

        # # train transition model
        # transition_model = TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1)
        # # transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
        # #                                                                 f'transition_model_0_16.pth')))
        # save_dir = os.path.join(cfg.train.save_dir, 'acrobot_models', f'transition_model_0.pth')
        # train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)

        for i in range(1, len(contexts)):
        # i = 8
            env.context_id = i
            # render = lambda: plt.imshow(env.render())
            env.reset()
            print(f"Currently using {param}: ", env.context[param])
            # policy = PPO.load(cfg.train.load_dir)
            # trajectories = collect_trajectories(env, policy, episodes=cfg.train.collect_episodes)
            # obs, actions, delta_obs = trajs_to_trans(trajectories)
            obs, actions, delta_obs = collect_transitions(env, policy, num_steps=cfg.train.collect_steps)
            # use model ensemble
            # models = [TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1) for _ in range(cfg.train.model_num)]
            transition_model = TransitionModel(obs_dim=env.observation_space['obs'].shape[0], action_dim=1)
            transition_model.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
                                                                            f'transition_model_0.pth')))
            save_dir = os.path.join(cfg.train.save_dir, 'acrobot_models', param, f'transition_model_{i}.pth')
            train_transition_model(transition_model, obs, actions, delta_obs, save_dir=save_dir)


if __name__ == '__main__':
    main()