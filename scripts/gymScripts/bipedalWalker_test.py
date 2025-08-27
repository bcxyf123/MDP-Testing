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

from gymEnvGen import *
from utils.utils import t_paired_test

rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])

def split_train_test_episodes(episodes, test_split=0.1, shuffle=True):
    if shuffle:
        np.random.shuffle(episodes)
    i_split = int(len(episodes) * (1 - test_split))
    train_episodes = episodes[:i_split]
    test_episodes = episodes[i_split:]
    return train_episodes, test_episodes

def CARL_obs_transform(obs):
    if isinstance(obs, dict):
        return obs['obs']
    if isinstance(obs, tuple):
        return obs[0]['obs']
    if isinstance(obs, np.ndarray):
        return obs

def policy_rollout_fixed(env, policy, max_steps=200):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state = env.reset()
    state = CARL_obs_transform(state)
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    step = 0
    while(step < max_steps):
        action, _ = policy.predict(state, deterministic=True)
        n_state, reward, done, _truncated, _info = env.step(action)
        n_state = CARL_obs_transform(n_state)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state.copy()
        dones.append(done)
        step += 1
        if done:
            state = env.reset()
            state = CARL_obs_transform(state)

    return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)


def policy_rollout(env, policy, max_steps=200):
    states, actions, rewards, dones = (
        [],
        [],
        [],
        [],
    )
    state = env.reset()
    state = CARL_obs_transform(state)
    states.append(state)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    while not done:
        action, _ = policy.predict(state, deterministic=True)
        n_state, reward, done, _truncated, _info = env.step(action)
        n_state = CARL_obs_transform(n_state)
        states.append(n_state)
        actions.append(action)
        rewards.append(reward)
        state = n_state.copy()
        dones.append(done)

    return np.array(states), np.array(actions).reshape(len(actions), -1), np.array(rewards), np.array(dones)


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
        detector = RIQN_Detector(env.env, model_kwargs={'gru_units': 8, 'quantile_embedding_dim': 32, 'num_quantile_sample': 8, 'device': 'cpu'})
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

def detect_powers(default_env, test_env, policy, detector, num_trajs):
    
    powers = []
    for _ in range(num_trajs):
        states_0, actions_0, _, _ = policy_rollout_fixed(default_env, policy)
        states_0 = np.array(states_0)
        actions_0 = np.array(actions_0)
        
        states_1, actions_1, _, _ = policy_rollout_fixed(test_env, policy)
        states_1 = np.array(states_1)
        actions_1 = np.array(actions_1)

        default_outputs = detector._predict_scores(states_0, actions_0)
        test_outputs = detector._predict_scores(states_1, actions_1)

        power = t_paired_test(default_outputs, test_outputs)
    powers.append(power)

    print(f'power: {np.mean(powers):.3f} +- {np.std(powers):.3f}')
    
    return np.mean(powers)

def Parameter_name_list(env_name):
    if 'CartPole' in env_name:
        param_list = ['MASSPOLE', 'LENGTH', 'FORCE_MAG', 'CART_MASS', 'GRAVITY']
        policy_name = 'ppo_cartpole.zip'
    elif 'Acrobot' in env_name:
        param_list = ['LINK_LENGTH_1', 'LINK_LENGTH_2', 'LINK_MASS_1', 'LINK_MASS_2', 'MAX_VEL_1', 'MAX_VEL_2', 'torque_noise_max']
        policy_name = 'ppo_acrobot.zip'
    elif 'MountainCar' in env_name:
        param_list = ['min_position', 'max_position', 'max_speed', 'goal_position', 'force', 
                'gravity', 'start_position', 'start_position_std', 'start_velocity', 'start_velocity_std']
        policy_name = 'MountainCar.zip'
    elif 'LunarLander' in env_name:
        param_list = ['FPS', 'SCALE']
        policy_name = 'LunarLander.zip'
    elif 'BipedalWalker' in env_name:
        param_list = ['fps', 'scale', 'gravity_x', 'gravity_y', 'friction', 'terrain_step', 'terrain_length', 'terrain_height', 'terrain_grass', 'terrain_startpad', 'motors_torque', 'speed_hip', 'speed_knee', 'lidar_range', 'leg_down', 'leg_w', 'leg_h', 'initial_random', 'viewport_w', 'viewport_h']
        policy_name = 'BipedalWalker.zip'
    elif 'CarRacing' in env_name:
        param_list = ['VEHICLE']
        policy_name = 'CarRacing.zip'
    else:
        raise ValueError(f'Unknown environment: {env_name}')
    
    return param_list, policy_name

def genEnv(env_name, param_name):
    if 'CartPole' in env_name:
        env, contexts, default_id = generate_CartPoleEnvs(parameter=param_name)
    elif 'Acrobot' in env_name:
        env, contexts, default_id = generate_AcrobotEnvs(parameter=param_name)
    elif 'MountainCar' in env_name:
        env, contexts, default_id = generate_mountaincarEnvs(parameter=param_name)
    elif 'LunarLander' in env_name:
        env, contexts, default_id = generate_LunarLanderEnvs(parameter=param_name)
    elif 'BipedalWalker' in env_name:
        env, contexts, default_id = generate_bipedalwalkerEnvs(parameter=param_name)
    elif 'CarRacing' in env_name:
        env, contexts, default_id = generate_VehicleRacingEnvs(parameter=param_name)
    else:
        raise ValueError(f'Unknown environment: {env_name}')
    
    return env, contexts, default_id

def single_test():
    env_name = "BipedalWalker"
    param_name = "LINK_MASS_2"
    params_name_list, policy_name = Parameter_name_list(env_name)
    policy = PPO.load(os.path.join("C:/Users/taizun/Desktop/OOD/scripts/gymScripts/models", policy_name))

    env_0, contexts, default_id = genEnv(env_name, param_name)
    env_1, _, _ = genEnv(env_name, param_name)
    env_0.context_id = 0
    env_0.reset()

    # detector_name = 'riqn'
    for detector_name in ['pedm', 'gmm', 'gaussian', 'knn', 'forest', ]:
    # for detector_name in ['riqn', ]:
        print(f"Currently using {detector_name}: ")
        detector = train_detector(env_0, policy, detector_type=detector_name, collect_episodes=500)
        print('training finished')
        powers = []
        for i in range(1, len(contexts)):
            env_1.context_id = i
            env_1.reset()
            print(f"Currently using {env_name} {detector_name} {param_name}: ", env_1.context[param_name])
            print(f"model {i}/{len(contexts)-1}")
            power = detect_powers(env_0, env_1, policy, detector, num_trajs=10)
            powers.append(power)
            print()
        
        save_dir = f'tables/{env_name}/{param_name}/{detector_name}.csv'
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        data = {
            'powers': powers
        }
        df = pd.DataFrame(data)
        df.to_csv(save_dir, index=False)

@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    
    env_name = 'Acrobot'
    params, policy_name = Parameter_name_list(env_name)
    policy = PPO.load(os.path.join(cfg.train.load_dir, policy_name))
    for param_name in params:
    # param_name = 'torque_noise_max'
        env_0, contexts, default_id = genEnv(env_name, param_name)
        env_1, _, _ = genEnv(env_name, param_name)
        env_0.context_id = 0
        env_0.reset()

        # detector_name = 'riqn'
        for detector_name in ['pedm', 'gmm', 'gaussian', 'knn', 'forest', ]:
        # for detector_name in ['riqn', ]:
            print(f"Currently using {detector_name}: ")
            detector = train_detector(env_0, policy, detector_type=detector_name, collect_episodes=cfg.train.collect_episodes)
            print('training finished')
            powers = []
            for i in range(1, len(contexts)):
                env_1.context_id = i
                env_1.reset()
                print(f"Currently using {env_name} {detector_name} {param_name}: ", env_1.context[param_name])
                print(f"model {i}/{len(contexts)-1}")
                power = detect_powers(env_0, env_1, policy, detector, num_trajs=10)
                powers.append(power)
                print()
            
            save_dir = f'tables/{env_name}/{param_name}/{detector_name}.csv'
            if not os.path.exists(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))

            data = {
                'powers': powers
            }
            df = pd.DataFrame(data)
            df.to_csv(save_dir, index=False)

if __name__ == '__main__':
    # main()
    single_test()
