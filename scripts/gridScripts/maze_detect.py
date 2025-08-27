import os
import sys
sys.path.append('../..')
import numpy as np
from collections import namedtuple
from stable_baselines3 import PPO
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import pandas as pd

from algo.pedm.ood_baselines.classical.gmm_detector import GMM_Detector
from algo.pedm.ood_baselines.classical.guassian_detector import GAUSSIAN_Detector
from algo.pedm.ood_baselines.classical.knn_detector import KNN_Detector
from algo.pedm.ood_baselines.classical.isolation_forest import ISOFOREST_Detector
from algo.pedm.pedm.pedm_detector import PEDM_Detector
from algo.pedm.ood_baselines.riqn.riqn_detector import RIQN_Detector

from envs.gym_simplegrid.generator import genEnv
from utils.utils import *

rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])

def split_train_test_episodes(episodes, test_split=0.1, shuffle=True):
    if shuffle:
        np.random.shuffle(episodes)
    i_split = int(len(episodes) * (1 - test_split))
    train_episodes = episodes[:i_split]
    test_episodes = episodes[i_split:]
    return train_episodes, test_episodes

def policy_rollout(env, policy, max_steps=200):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state, _ = env.reset()
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action, _ = policy[state]
        n_state, reward, done, _truncated, _info = env.step(action)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state.copy()
        dones.append(done)

    return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)


def policy_rollout_fixed(env, policy, max_steps=50):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state, _ = env.reset()
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    step = 0
    while(step < max_steps):
        action = policy[state]
        n_state, reward, done, _truncated, _info = env.step(action)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state
        dones.append(done)
        step += 1
        if done:
            state, _ = env.reset()

    return np.array(states).reshape(len(states), -1), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)



def train_detector(env, policy, detector_type, collect_episodes):
    if 'pedm' in detector_type:
        if 'error_samples' in detector_type:
            detector = PEDM_Detector(env, criterion='pred_error_samples')
        elif 'std_samples' in detector_type:
            detector = PEDM_Detector(env, criterion='pred_std_samples')
        elif 'pdf' in detector_type:
            detector = PEDM_Detector(env, criterion='pred_error_pdf')
        elif 'p_value' in detector_type:
            detector = PEDM_Detector(env, criterion='p_value')
        else:
            detector = PEDM_Detector(env,)
    elif 'riqn' in detector_type:
        detector = RIQN_Detector(env, horizon=5, model_kwargs={'gru_units': 8, 'quantile_embedding_dim': 32, 'num_quantile_sample': 8, 'device': 'cpu'})
    elif 'gmm' in detector_type:
        detector = GMM_Detector(env)
    elif 'gaussian' in detector_type:
        detector = GAUSSIAN_Detector(env)
    elif 'knn' in detector_type:
        detector = KNN_Detector(env)
    elif 'forest' in detector_type:
        detector = ISOFOREST_Detector(env)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    ep_data = []
    trange = range(collect_episodes)
    for _ in trange:
        ep_data.append(rollout(*policy_rollout_fixed(env, policy)))
    train_ep_data, val_ep_data = split_train_test_episodes(ep_data, test_split=0.1, shuffle=True)
    detector._fit(train_ep_data, val_ep_data)

    return detector


def collect_transitions(env, policy, num_steps):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )

    state, _ = env.reset()
    # state = CARL_obs_transform(state)
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    for _ in range(num_steps):
        action = policy[state]
        n_state, reward, done, _truncated, _info = env.step(action)
        # n_state = CARL_obs_transform(n_state)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state
        dones.append(done)

    return np.array(states).reshape(len(states), -1), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)


def detect_powers(test_env, policy, detector, num_trajs):
    
    powers = []
    outputs = []
    for _ in range(num_trajs):

        states_1, actions_1, _, _ = collect_transitions(test_env, policy, num_steps=50)
        # states_1 = np.array(states_1)
        # actions_1 = np.array(actions_1)

        test_outputs = detector._predict_scores(states_1, actions_1)
        outputs.append(test_outputs)

    outputs = np.array(outputs).reshape(-1)
    power = np.sum(outputs > detector.threshold) / len(outputs)
    print(f"Power: {power:.3f}")
    # print(f"Average power: {np.mean(powers):.3f} +- {np.std(powers):.3f}")
    
    return power


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    
    # load policy
    policy_dir = f'save/model/policy/map_0_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]
    default_env = genEnv(map_size=(4,8), map_id=0, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    prob_list = np.arange(0.9, 0.1, -0.01)
    map_list = range(7)

    # for detector_name in ['pedm', 'gmm', 'gaussian', 'knn', 'forest', 'riqn']:
    for detector_name in ['pedm']:
    # for detector_name in ['pedm_pdf']:
        detector = train_detector(default_env, policy, detector_type=detector_name, collect_episodes=50)
        powers = []
        for map_id in map_list:
        # for p in prob_list:
            # test_env = genEnv(map_size=(4,8), map_id=0, max_prob=p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
            test_env = genEnv(map_size=(4,8), map_id=map_id, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
            print(f"Currently using {detector_name} {map_id}: ")
            # print(f"Currently using {detector_name} {p}: ")
            power = detect_powers(test_env, policy, detector, num_trajs=100)
            powers.append(power)
            print()
            
        save_dir = f'tables/map_d_{detector_name}.csv'
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        data = {
            'powers': powers
        }
        df = pd.DataFrame(data)
        df.to_csv(save_dir, index=False)

def self_detect():
    # load policy
    policy_dir = f'save/model/policy/map_0_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]
    default_env = genEnv(map_size=(4,8), map_id=0, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    for detector_name in ['pedm', 'gmm', 'gaussian', 'knn', 'forest', ]:
        print(f"Currently using {detector_name}: ")
        detector = train_detector(default_env, policy, detector_type=detector_name, collect_episodes=100)
        powers = []
        t_stats = []
        p_values = []

        # for i in range(10):
        test_env = genEnv(map_size=(4,8), map_id=0, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
        power = detect_powers(test_env, policy, detector, num_trajs=100)
        # powers.append(power)
        # t_stats.append(t_stat)
        # p_values.append(p_value)
        
        # print(f"powers: {np.mean(powers):.2f} +- {np.std(powers):.2f}")
        # print(f"t_stats: {np.mean(t_stats):.2f} +- {np.std(t_stats):.2f}")
        # print(f"p_values: {np.mean(p_values):.2f} +- {np.std(p_values):.2f}")
        print()

if __name__ == '__main__':
    
    # self_detect()
    main()
