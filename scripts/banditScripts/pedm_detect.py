import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

# from utils import *
from train_transition_model import *
from statsmodels.stats.power import TTestIndPower

import pandas as pd
from statsmodels.multivariate.manova import MANOVA

def perform_manova(data1, data2, n_features):
    # stack data1 and data2 together to create a DataFrame
    data = np.vstack([data1, data2])
    labels = np.array([0]*len(data1) + [1]*len(data2))  # 0 for data1, 1 for data2

    df = pd.DataFrame(data)
    df['label'] = labels

    # dynamically construct the formula for MANOVA
    formula = 'label ~ 0 + ' + ' + '.join(f'C{i}' for i in range(n_features))

    maov = MANOVA.from_formula(formula, data=df)
    print(maov.mv_test())

from statsmodels.stats.power import FTestAnovaPower
from scipy import stats

def calculate_power(data1, data2, n_features, alpha=0.05):
    # calculate the sample size and the number of groups
    n_samples = len(data1) + len(data2)
    n_groups = 2

    powers = []

    for i in range(n_features):
        # perform ANOVA for each feature
        f_statistic, p_value = stats.f_oneway(data1[:, i], data2[:, i])

        # calculate the effect size
        effect_size = np.sqrt(f_statistic / (n_samples - n_groups))

        # calculate the power
        power = FTestAnovaPower().solve_power(effect_size, nobs=n_samples, alpha=alpha, k_groups=n_groups)
        powers.append(power)

    # print(f'mean powers: {np.mean(powers)}')
    # return the average power
    return np.mean(powers)

def model_ensemble(n_models, obs_dim, action_dim=1):
    models = []
    for i in range(n_models):
        model = TransitionModel(obs_dim=obs_dim, action_dim=action_dim)
        model.load_state_dict(torch.load(f'transition_model_{i}.pth'))
        models.append(model)
    return models

def predict(models, solver, num_trajs, device="cpu"):
    
    inputs, _, outputs = collect_trajectories(solver, num_trajs)
    powers = []
    for actions, rewards in zip(inputs, outputs):
        model_predicts = []
        states = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        delta_states = np.array(rewards)

        for model in models:
            outputs = model.sample(states, actions)
            model_predicts.append(outputs)
        model_predicts = torch.mean(torch.stack(model_predicts), dim=0).cpu().detach().numpy()

        power = calculate_power(delta_states, model_predicts, delta_states.shape[1])
        powers.append(power)
    # print(f'ensemble differences: {np.mean(differences):.3f}')
    print(f'power: {np.mean(powers):.3f}')
    
    return np.mean(powers)
    

@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    default_bandit = BernoulliBandit(p=0.1)
    default_solver = ThompsonSampling(default_bandit)
    default_solver.policy_fixed = True
    models = model_ensemble(cfg.train.model_num, obs_dim=1, action_dim=1)
    power = predict(models, default_solver, num_trajs=cfg.detect.collect_trajs, device='cpu')

    a = deepcopy(default_solver._a)
    b = deepcopy(default_solver._b)
    powers = []
    p_list = np.arange(0.1, 1.0, 0.005)
    for p in p_list:
        print('bandit probability: ', p)
        bandit = BernoulliBandit(p=p)
        solver = ThompsonSampling(bandit)
        solver._a = a
        solver._b = b
        solver.policy_fixed = True
        power = predict(models, solver, num_trajs=cfg.detect.collect_trajs, device='cpu')
        powers.append(power)
        print()
    
    data = {
        'powers': powers
    }
    df = pd.DataFrame(data)
    df.to_csv(f'tables/pedm.csv', index=False)

    # plot
    df = pd.read_csv(f'tables/pedm.csv')
    powers = df['powers']

    plt.figure()
    plt.plot(p_list, powers, label='Power')
    plt.scatter(p_list[0], powers[0], color='red', label='default power')
    plt.xlabel(f'arm probability')
    plt.ylabel('Power')
    plt.legend()
    plt.savefig(f'figs/pdf/pedm_power.pdf', dpi=300)
    plt.savefig(f'figs/png/pedm_power.png', dpi=300)

if __name__ == '__main__':
    main()
            