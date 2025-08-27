import sys
sys.path.append('../..')
import gym
import numpy as np
import hydra
import random
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import *
from scripts.gymScripts.gymEnvGen import generate_AcrobotEnvs, generate_CartPoleEnvs
from train_transition_model import CARL_obs_transform

def evaluate_policy(env, policy, num_episodes=5, gamma=0.99, render=False):
    returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        obs = CARL_obs_transform(obs)
        episode_return = 0
        done = False
        truncated = False
        step = 0
        while not (done or truncated):
            step += 1
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            obs = CARL_obs_transform(obs)
            episode_return += gamma**step *reward
            # if render:
            #     env.render()
        returns.append(episode_return)
    return returns

@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):

    np.random.seed(1)

    policy = PPO.load(cfg.train.load_dir)
    env, contexts = generate_AcrobotEnvs(cfg.detect.param_name)
    total_value_list = []

    x_list = []
    powers = []
    mean_values = []
    std_values = []

    # default env
    env.context_id = 0
    env.reset()
    returns_0 = evaluate_policy(env, policy, num_episodes=cfg.detect.collect_trajs,)

    for j in range(1, len(contexts)):
        env.context_id = j
        env.reset()
        print(f"Currently using : {cfg.detect.param_name}", env.context[cfg.detect.param_name])
        returns_1 = evaluate_policy(env, policy, num_episodes=cfg.detect.collect_trajs,)
        power = t_ind_test(returns_0, returns_1)
        powers.append(power)
        mean_values.append(np.mean(returns_1))
        std_values.append(np.std(returns_1))
        x_list.append(env.context[cfg.detect.param_name])

    print(mean_values, std_values)
    print('power: ', powers)

    # total_value_list.append(value_list)
    # x_arr = np.array(x_list)
    # total_value_arr = np.array(total_value_list)
    # data = np.vstack((x_arr, total_value_arr))
    # df = pd.DataFrame(data)
    # df.to_csv(f'models/acrobot_models/policy_value_{cfg.detect.param_name}.csv', header=None, index=None)

    # arr = np.array(pd.read_csv(f'models/acrobot_models/policy_value_{cfg.detect.param_name}.csv', header=None))
    # x_arr = arr[0, :]
    # value_arr = arr[1:, :]
    # df = pd.DataFrame(value_arr).T

    # mean_values = df.mean(axis=1)
    # std_values = df.std(axis=1)

    x_arr = np.array(x_list)
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)

    plt.figure()
    plt.plot(x_arr, mean_values)
    plt.annotate('default', (1.0, mean_values[2]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)

    plt.xlabel(cfg.detect.param_name)
    plt.ylabel('return')
    plt.title(f'returns of different {cfg.detect.param_name}')
    plt.savefig(f'models/acrobot_models/return_{cfg.detect.param_name}.png')

    plt.figure()
    plt.plot(x_list, powers)
    plt.annotate('default', (1.0, powers[2]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel(cfg.detect.param_name)
    plt.ylabel('power')
    plt.title(f'power of values')
    plt.savefig(f'models/acrobot_models/value_power_{cfg.detect.param_name}.png')


if __name__ == '__main__':

    main()

    # plt.figure()
    # plt.plot(value_list, marker='o')
    # plt.annotate('default', (9, value_list[9]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.savefig('models/acrobot_models/policy evaluation.png')

