import sys
sys.path.append("../..")
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils import *
from envs.gym_simplegrid.generator import *
from train_forward_model import collect_trajectories, TransitionModel


def self_detect(sample_size=50):
    
    ratios = []
    policy_dir = 'save/model/policy/map_0_policy_2000.csv'
    default_model_dir = 'save/model/forward_model/forward_model_default.pth'
    test_model_dir = 'save/model/forward_model/forward_model_0.60.pth'
    policy = read_csv(policy_dir)[:, -1]
    default_model = TransitionModel(obs_dim=1, action_dim=1)
    default_model.load_state_dict(torch.load(default_model_dir))
    test_model = TransitionModel(obs_dim=1, action_dim=1)
    test_model.load_state_dict(torch.load(test_model_dir))
    # mdp = genMDP(map_size=(4,8), max_prob=0.6, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    env = genEnv(map_size=(4,8), max_prob=0.6, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    trajectories = collect_trajectories(env, policy, sample_size)
    model0 = default_model
    model1 = test_model
    ratios = []
    likelihood0_list = []
    likelihood1_list = []
    for trj in trajectories:
        likelihood0, likelihood1, ratio = likelihoodRatio(trj, model0, model1)
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratios.append(ratio)

    power = chi_test(ratios, df=5)
    
    # # plot
    # plt.figure()
    # plt.hist(likelihood0_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='likelihood 0')
    # plt.xlabel('likelihood 0')
    # plt.ylabel('frequency')
    # plt.title('likelihood 0')
    # plt.legend()
    # plt.savefig(f'likelihood0_{sample_size}.png')

    # plt.figure()
    # plt.hist(likelihood1_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='likelihood 1')
    # plt.xlabel('likelihood 1')
    # plt.ylabel('frequency')
    # plt.title('likelihood 1')
    # plt.legend()
    # plt.savefig(f'likelihood1_{sample_size}.png')

    # plt.figure()
    # plt.hist(ratios, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='likelihood ratio')
    # plt.xlabel('likelihood ratio')
    # plt.ylabel('frequency')
    # plt.title('likelihood ratio')
    # plt.legend()
    # plt.savefig(f'likelihood_ratio_{sample_size}.png')

    print(f'likelihood0: {np.mean(likelihood0_list):.3f}, {np.std(likelihood0_list):.3f}')
    print(f'likelihood1: {np.mean(likelihood1_list):.3f}, {np.std(likelihood1_list):.3f}')
    print(f'ratio: {np.mean(ratios):.3f}, {np.std(ratios):.3f}')
    print(f'power: {power:.2f}')


def trajectory_test(test_p=0.60, sample_size=50):
    
    policy_dir = 'save/model/policy/map_0_policy_2000.csv'
    default_model_dir = 'save/model/forward_model/forward_model_default.pth'
    str_p = format(test_p, '.2f')
    test_model_dir = f'save/model/forward_model/forward_model_{str_p}.pth'

    policy = read_csv(policy_dir)[:, -1]
    default_model = TransitionModel(obs_dim=1, action_dim=1)
    default_model.load_state_dict(torch.load(default_model_dir))
    test_model = TransitionModel(obs_dim=1, action_dim=1)
    test_model.load_state_dict(torch.load(test_model_dir))
    
    env = genEnv(map_size=(4,8), max_prob=test_p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    trajectories = collect_trajectories(env, policy, sample_size)
    model0 = default_model
    model1 = test_model

    likelihood0_list = []
    likelihood1_list = []
    ratios = []

    for trj in trajectories:
        likelihood0, likelihood1, ratio = likelihoodRatio(trj, model0, model1)
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratios.append(ratio)
    
    power = chi_test(ratios, df=5)

    print(f'likelihood0: {np.mean(likelihood0_list):.3f}, {np.std(likelihood0_list):.3f}')
    print(f'likelihood1: {np.mean(likelihood1_list):.3f}, {np.std(likelihood1_list):.3f}')
    print(f'ratio: {np.mean(ratios):.3f}, {np.std(ratios):.3f}')
    print(f'power: {power:.2f}')

    return likelihood0_list, likelihood1_list, ratios, power


def ratio_to_power():
    data = np.array(pd.read_csv('save\detect\likelihood ratio.csv', header=None))[1:, :]
    powers = []
    for idx in range(7):
        ratio_arr = data[:, idx]
        power = chi_test(ratio_arr)
        powers.append(power)
    
    plt.plot(range(7), powers)
    plt.annotate('default', (0, powers[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)
    plt.xlabel('Test Map ID')
    plt.ylabel('Power')
    plt.title(f'Power of Different Maps')
    plt.savefig('likelihood power.png')
    
# def main(policy_id=0, map_id=0, prob_id=0, episodes=50, policy_path=None, model_path=None):
#     # load trained policy
#     if policy_path is None:
#         policy_path = f'save/model/policy/map_{policy_id}_policy_2000.csv'
#     if model_path is None:
#         model_path = f'scripts/save/model/forward_model/map_0_forward_model.pt'
#     policy = read_csv(policy_path)[:, -1]
#     # generate test environment
#     mdp = genMDP(map_size=(4,8), map_id=policy_id, prob_id=prob_id, start_loc=(3,0), goal_loc=(0,7), deterministic=False, seed=222333)
#     env = genEnv(map_size=(4,8), map_id=map_id, prob_id=prob_id, start_loc=(3,0), goal_loc=(0,7), deterministic=False, seed=222333)
#     # compute likelihood
#     trajectories = collect_trajectory(policy, env, episodes)
#     traj_likelihood_list_0 = []
#     traj_likelihood_list_1 = []
#     for trj in trajectories:
#         traj_likelihood_list_0.append(compute_likelihood(trj, mdp))
#         traj_likelihood_list_1.append(compute_likelihood(trj, env))
#     likelihood_0 = np.mean(traj_likelihood_list_0)
#     likelihood_1 = np.mean(traj_likelihood_list_1)
#     likelihood_ratio, is_null = likelihoodRatioTest(traj_likelihood_list_0, traj_likelihood_list_1, c=0.5, alpha=0.05)

#     print(likelihood_0, likelihood_1, likelihood_ratio, is_null)

#     # # compute likelihood and rewards
#     # data = collect_data(policy, env, episodes)
#     # likelihood= compute_likelihood(data, mdp)
#     # # likelihood_ratio, is_null = likelihoodRatioTest(likelihood, 1, c=0.5, alpha=0.05)
#     # rew_diff = compute_rews(data, mdp)
#     # record = {'policy_id': policy_id, 'test_env': prob_id, 'likelihood': likelihood, 'rew_diff': rew_diff}
#     # # save results
#     # write_csv(record, save_path)

#     return likelihood_0, likelihood_1, likelihood_ratio, is_null

if __name__=='__main__':
    # for sample_size in [10, 20, 50, 100, 200, 500]:
    #     self_detect(sample_size)

    prob_list = np.arange(0.10, 1.0, 0.01)
    likelihood0_means = []
    likelihood0_stds = []
    likelihood1_means = []
    likelihood1_stds = []
    ratio_means = []
    ratio_stds = []
    powers = []

    for p in prob_list:
        p = round(p, 2)
        print(f'p={p}')
        likelihood0_list, likelihood1_list, ratios, power = trajectory_test(p, sample_size=100)

        likelihood0_means.append(np.mean(likelihood0_list))
        likelihood0_stds.append(np.std(likelihood0_list))
        likelihood1_means.append(np.mean(likelihood1_list))
        likelihood1_stds.append(np.std(likelihood1_list))
        ratio_means.append(np.mean(ratios))
        ratio_stds.append(np.std(ratios))
        powers.append(power)
    
    # plot
    plt.figure()
    plt.plot(prob_list, likelihood0_means, label='likelihood 0')
    plt.fill_between(prob_list, np.array(likelihood0_means)-np.array(likelihood0_stds), np.array(likelihood0_means)+np.array(likelihood0_stds), alpha=0.2)
    plt.plot(prob_list, likelihood1_means, label='likelihood 1')
    plt.fill_between(prob_list, np.array(likelihood1_means)-np.array(likelihood1_stds), np.array(likelihood1_means)+np.array(likelihood1_stds), alpha=0.2)
    plt.xlabel('Test Environment Probability')
    plt.ylabel('Likelihood')
    plt.title('Likelihood of Different Test Environments')
    plt.legend()
    plt.savefig('likelihood.png')

    plt.figure()
    plt.plot(prob_list, ratio_means, label='likelihood ratio')
    plt.fill_between(prob_list, np.array(ratio_means)-np.array(ratio_stds), np.array(ratio_means)+np.array(ratio_stds), alpha=0.2)
    plt.xlabel('Test Environment Probability')
    plt.ylabel('Likelihood Ratio')
    plt.title('Likelihood Ratio of Different Test Environments')
    plt.legend()
    plt.savefig('likelihood_ratio.png')

    plt.figure()
    plt.plot(prob_list, powers, label='power')
    plt.xlabel('Test Environment Probability')
    plt.ylabel('Power')
    plt.title('Power of Different Test Environments')
    plt.legend()
    plt.savefig('power.png')