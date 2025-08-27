import os
import gymnasium as gym
from gymnasium import spaces
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from brax import envs
from carl.envs import CARLBraxAnt, CARLAcrobot

from train_transition_model import CARL_obs_transform


def evaluate(env, model, num_episodes, render):
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        truncated = False
        total_reward = 0
        step = 0

        while not done and step < 500:
            step += 1
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if render:
                env.render()

        total_rewards.append(total_reward)
        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

    average_reward = sum(total_rewards) / num_episodes
    print(f'\nAverage Reward over {num_episodes} episodes: {average_reward}')
    return average_reward


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Usage
    env = gym.make(cfg.gym_env.id)
    # train
    if cfg.train.train_mode:
        tensorboard_log = os.path.join(cfg.train.save_dir, cfg.train.env_name)
        model = PPO(cfg.ppo_params.policy, env, device=cfg.ppo_params.device, 
                    verbose=cfg.ppo_params.verbose, tensorboard_log=tensorboard_log)
        model.learn(total_timesteps=cfg.train.total_timesteps)
        save_dir = os.path.join(cfg.train.save_dir, cfg.train.env_name)
        model.save(save_dir)

    # test
    # policy = PPO.load(cfg.train.load_dir)
    # evaluate(env, policy, cfg.train.num_eval_episodes, cfg.train.render, )

if __name__ == '__main__':
    main()
