import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hydra
from omegaconf import DictConfig
from copy import deepcopy
from collections import namedtuple

from algo.pedm.ood_baselines.classical.gmm_detector import GMM_Detector
from algo.pedm.ood_baselines.classical.guassian_detector import GAUSSIAN_Detector
from algo.pedm.ood_baselines.classical.knn_detector import KNN_Detector
from algo.pedm.ood_baselines.classical.isolation_forest import ISOFOREST_Detector
from algo.pedm.pedm.pedm_detector import PEDM_Detector
from algo.pedm.ood_baselines.riqn.riqn_detector import RIQN_Detector

from envs.bandit.bandit import BernoulliBandit
from solverScript import ThompsonSampling

# from pedm_detect import *
# from utils.utils import t_paired_test

rollout = namedtuple("rollout", ["states", "actions",])

def split_train_test_episodes(episodes, test_split=0.1, shuffle=True):
    if shuffle:
        np.random.shuffle(episodes)
    i_split = int(len(episodes) * (1 - test_split))
    train_episodes = episodes[:i_split]
    test_episodes = episodes[i_split:]
    return train_episodes, test_episodes

def policy_rollout(solver, update=False):
    solver.run(500, update)
    states = solver.actions
    actions = solver.rewards
    solver.reset()
    states = np.array(states).reshape(len(states), -1)
    actions = np.array(actions).reshape(len(actions), -1)
    states = np.concatenate([states, actions], axis=1)
    

    return states, actions 


def train_detector(solver, detector_type, collect_episodes):
    if 'pedm' in detector_type:
        if 'error_samples' in detector_type:
            detector = PEDM_Detector(solver.bandit, criterion='pred_error_samples')
        elif 'std_samples' in detector_type:
            detector = PEDM_Detector(solver.bandit, criterion='pred_std_samples')
        elif 'pdf' in detector_type:
            detector = PEDM_Detector(solver.bandit, criterion='pred_error_pdf')
        elif 'p_value' in detector_type:
            detector = PEDM_Detector(solver.bandit, criterion='p_value')
        else:
            detector = PEDM_Detector(solver.bandit,)
    elif 'riqn' in detector_type:
        detector = RIQN_Detector(solver.bandit, model_kwargs={'gru_units': 8, 'quantile_embedding_dim': 32, 'num_quantile_sample': 8, 'device': 'cpu'})
    elif 'gmm' in detector_type:
        detector = GMM_Detector(solver.bandit)
    elif 'gaussian' in detector_type:
        detector = GAUSSIAN_Detector(solver.bandit)
    elif 'knn' in detector_type:
        detector = KNN_Detector(solver.bandit)
    elif 'forest' in detector_type:
        detector = ISOFOREST_Detector(solver.bandit)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    ep_data = []
    trange = range(collect_episodes)
    for _ in trange:
        ep_data.append(rollout(*policy_rollout(solver)))
    train_ep_data, val_ep_data = split_train_test_episodes(ep_data, test_split=0.1, shuffle=True)
    detector._fit(train_ep_data, val_ep_data)

    return detector


def detect_powers(default_env, test_env, detector, num_trajs):
    
    powers = []
    outputs = []
    for _ in range(num_trajs):
        # states_0, actions_0 = policy_rollout(default_env, )
        # states_0 = np.array(states_0)
        # actions_0 = np.array(actions_0)
        
        states_1, actions_1 = policy_rollout(test_env, )
        states_1 = np.array(states_1)
        actions_1 = np.array(actions_1)

        # default_outputs = detector._predict_scores(states_0, actions_0)

        test_outputs = detector._predict_scores(states_1, actions_1)
        outputs.append(test_outputs)

    outputs = np.array(outputs).reshape(-1)
    power = np.sum(outputs > detector.threshold) / len(outputs)
        # power = t_paired_test(default_outputs, test_outputs)
    # powers.append(power)
    print(power)
    # print(f'power: {np.mean(powers):.3f} +- {np.std(powers):.3f}')
    
    return power


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):

    for detector_type in ['pedm', 'gmm', 'gaussian', 'knn', 'forest',]:
        
        default_bandit = BernoulliBandit(p=0.9)
        default_solver = ThompsonSampling(default_bandit)
        default_solver.run(500)
        default_solver.policy_fixed = True
        default_solver.reset()
        detector = train_detector(default_solver, detector_type, collect_episodes=100)

        a = deepcopy(default_solver._a)
        b = deepcopy(default_solver._b)

        p_list = np.arange(0.9, 0.1, -0.1)
        powers = []
        for p in p_list:
            print(f"arm probability {detector_type}:", p)
            test_bandit = BernoulliBandit(p=p)
            test_solver = ThompsonSampling(test_bandit,)
            test_solver._a = a
            test_solver._b = b
            test_solver.policy_fixed = True
            power = detect_powers(default_solver, test_solver, detector, num_trajs=10)
            powers.append(power)
            print()
        
        data = {
            'prob': p_list,
            'powers': powers
        }
        df = pd.DataFrame(data)
        df.to_csv(f'tables/new/{detector_type}.csv', index=False)

        # # plot
        # df = pd.read_csv(f'tables/{detector_type}.csv')
        # powers = df['powers']

        # plt.figure()
        # plt.plot(p_list, powers, label='Power')
        # # plt.scatter(p_list[0], powers[0], color='red', label='default power')
        # plt.xlabel(f'{cfg.detect.param_name}')
        # plt.ylabel('Power')
        # plt.legend()
        # plt.savefig(f'figs/pdf/{detector_type}_power.pdf', dpi=300)
        # plt.savefig(f'figs/png/{detector_type}_power.png', dpi=300)


if __name__ == '__main__':
    main()
