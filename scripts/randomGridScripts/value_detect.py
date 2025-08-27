import sys
sys.path.append('../..')
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import *
from envs.gym_randomgrid.mdps.grid_mdp import RandomGridEnv
from envs.gym_randomgrid.generator import MAP_LIB, genEnv, genMDP
from algo.dp_eval import dp_eval
from algo.value_eval import mc_eval


def value_test_1(default_map=0, test_map=0, default_prob=0.6, test_prob=0.6, policy_id=0, n_trials=10):
    # load policy
    policy_dir = f'save/model/policy/map_{policy_id}_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]

    # load env
    default_env = genEnv(map_size=(4,8), map_id=0, max_prob=default_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    test_env = genEnv(map_size=(4,8), map_id=0, max_prob=test_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    # eval
    returns_0 = compute_returns(policy, default_env, gamma=0.99, iterations=n_trials)
    returns_1 = compute_returns(policy, test_env, gamma=0.99, iterations=n_trials)
    return_diff = np.array(returns_0) - np.array(returns_1)
    power = t_paired_test(returns_0, returns_1)

    # plot
    plt.figure()
    plt.hist(returns_0, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0$')
    plt.xlabel('$G_0$')
    plt.ylabel('density')
    # # plt.title(f'$G_0$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns0_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns0_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(returns_1, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_1$')
    plt.xlabel('$G_1$')
    plt.ylabel('density')
    # # plt.title(f'$G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns1_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns1_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(return_diff, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0-G_1$')
    plt.xlabel('$G_0-G_1$')
    plt.ylabel('density')
    # # plt.title(f'$G_0-G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/return_diff_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/return_diff_{n_trials}.png', dpi=300)

    print(f'returns_0: {np.mean(returns_0):.3f}, {np.std(returns_0):.3f}')
    print(f'returns_1: {np.mean(returns_1):.3f}, {np.std(returns_1):.3f}')
    print(f'return_diff: {np.mean(return_diff):.3f}, {np.std(return_diff):.3f}')
    print(f'power: {power:.3f}')
    print()

    return returns_0, returns_1, return_diff, power


# def collect_states(env, policy, n_trials):
#     trajs = []
#     for _ in range(n_trials):
#         traj = []
#         state, _ = env.reset()
#         # traj.append(state)
#         while True:
#             action = policy[state]
#             next_state, reward, done, _, info = env.step(action)
#             traj.append(next_state)
#             state = next_state
#             if done:
#                 break
#         trajs.append(traj)
#     return trajs

def value_test_2(default_map=0, test_map=0, default_prob=0.6, test_prob=0.6, policy_id=0, gamma=0.99, n_trials=100):
    # load policy
    policy_dir = f'save/model/policy/map_{policy_id}_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]

    # load env
    default_env = genEnv(map_size=(4,8), map_id=0, max_prob=default_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    test_env = genEnv(map_size=(4,8), map_id=0, max_prob=test_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    # eval
    # compute returns 0
    returns_0 = compute_returns(policy, default_env, gamma=gamma, iterations=n_trials)
    # compute returns 1
    returns_1 = []
    for _ in range(n_trials):
        return_1 = 0
        state, _ = test_env.reset()
        default_env.reset()
        step = 0
        done = False
        while not done:
            action = policy[state]
            next_state, _, done, _, info = test_env.step(action)
            x, y = info['x'], info['y']
            reward = default_env.unwrapped.get_reward(x, y)
            return_1 += reward * (gamma ** step)
            state = next_state
            step += 1
        returns_1.append(return_1)
    
    return_diff = np.array(returns_0) - np.array(returns_1)
    power = t_paired_test(returns_0, returns_1)

    # plot
    plt.figure()
    plt.hist(returns_0, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0$')
    plt.xlabel('$G_0$')
    plt.ylabel('density')
    # plt.title(f'$G_0$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns0_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns0_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(returns_1, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_1$')
    plt.xlabel('$G_1$')
    plt.ylabel('density')
    # plt.title(f'$G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns1_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns1_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(return_diff, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0-G_1$')
    plt.xlabel('$G_0-G_1$')
    plt.ylabel('density')
    # plt.title(f'$G_0-G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/return_diff_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/return_diff_{n_trials}.png', dpi=300)

    print(f'returns_0: {np.mean(returns_0):.3f}, {np.std(returns_0):.3f}')
    print(f'returns_1: {np.mean(returns_1):.3f}, {np.std(returns_1):.3f}')
    print(f'return_diff: {np.mean(return_diff):.3f}, {np.std(return_diff):.3f}')
    print(f'power: {power:.3f}')
    print()

    return returns_0, returns_1, return_diff, power


def value_test_3(default_map=0, test_map=0, default_prob=0.6, test_prob=0.6, policy_id=0, gamma=0.99, n_trials=100):
    # load policy
    policy_dir = f'save/model/policy/map_{policy_id}_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]

    # load env
    default_env = genEnv(map_size=(4,8), map_id=0, max_prob=default_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    test_env = genEnv(map_size=(4,8), map_id=0, max_prob=test_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    # compute returns
    returns_0 = []
    returns_1 = []
    for _ in range(n_trials):
        return_0 = 0
        return_1 = 0
        state, _ = test_env.reset()
        default_env.reset()
        step = 0
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _, info = test_env.step(action)
            x, y = info['x'], info['y']
            reward_0 = default_env.unwrapped.get_reward(x, y)
            reward_1 = reward
            return_0 += reward_0 * (gamma ** step)
            return_1 += reward_1 * (gamma ** step)
            state = next_state
            step += 1
        returns_0.append(return_0)
        returns_1.append(return_1)
    
    return_diff = np.array(returns_0) - np.array(returns_1)
    power = t_paired_test(returns_0, returns_1)

    # plot
    plt.figure()
    plt.hist(returns_0, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0$')
    plt.xlabel('$G_0$')
    plt.ylabel('density')
    # plt.title(f'$G_0$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns0_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns0_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(returns_1, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_1$')
    plt.xlabel('$G_1$')
    plt.ylabel('density')
    # plt.title(f'$G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/returns1_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/returns1_{n_trials}.png', dpi=300)

    plt.figure()
    plt.hist(return_diff, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='$G_0-G_1$')
    plt.xlabel('$G_0-G_1$')
    plt.ylabel('density')
    # plt.title(f'$G_0-G_1$')
    plt.legend()
    plt.savefig(f'figs/pdf/return_diff_{n_trials}.pdf', dpi=300)
    plt.savefig(f'figs/png/return_diff_{n_trials}.png', dpi=300)

    print(f'returns_0: {np.mean(returns_0):.3f}, {np.std(returns_0):.3f}')
    print(f'returns_1: {np.mean(returns_1):.3f}, {np.std(returns_1):.3f}')
    print(f'return_diff: {np.mean(return_diff):.3f}, {np.std(return_diff):.3f}')
    print(f'power: {power:.3f}')
    print()

    return returns_0, returns_1, return_diff, power



def main():
    # sample_size_list = [10, 20, 50, 100, 200, 500]
    # for sample_size in sample_size_list:
    #     print('sample size: ', sample_size)
    #     value_test_3(policy_id=0, n_trials=sample_size)
    
    value_test_1(policy_id=0, n_trials=500)
    # test_id = 1

    # prob_list = np.arange(0.1, 1.0, 0.005)
    # return0_means = []
    # return0_stds = []
    # return1_means = []
    # return1_stds = []
    # return_diff_means = []
    # return_diff_stds = []
    # powers = []
    # for p in prob_list:
    #     print('test prob: ', p)
    #     returns_0, returns_1, return_diff, power = value_test_1(default_prob=0.6, test_prob=p, policy_id=0, n_trials=100)
    #     return0_means.append(np.mean(returns_0))
    #     return0_stds.append(np.std(returns_0))
    #     return1_means.append(np.mean(returns_1))
    #     return1_stds.append(np.std(returns_1))
    #     return_diff_means.append(np.mean(return_diff))
    #     return_diff_stds.append(np.std(return_diff))
    #     powers.append(power)

    # # plot
    # plt.figure()
    # plt.plot(prob_list, return0_means, label='$G_0$')
    # plt.fill_between(prob_list, np.array(return0_means)-np.array(return0_stds), np.array(return0_means)+np.array(return0_stds), alpha=0.2)
    # plt.plot(prob_list, return1_means, label='$G_1$')
    # plt.fill_between(prob_list, np.array(return1_means)-np.array(return1_stds), np.array(return1_means)+np.array(return1_stds), alpha=0.2)
    # plt.xlabel('reward probability $p$')
    # plt.ylabel('return')
    # # plt.title(f'return of different maps')
    # plt.legend()
    # plt.savefig(f'figs/pdf/return_{test_id}.pdf', dpi=300)
    # plt.savefig(f'figs/png/return_{test_id}.png', dpi=300)

    # plt.figure()
    # plt.plot(prob_list, return_diff_means, label='$G_0-G_1$')
    # plt.fill_between(prob_list, np.array(return_diff_means)-np.array(return_diff_stds), np.array(return_diff_means)+np.array(return_diff_stds), alpha=0.2)
    # plt.xlabel('reward probability $p$')
    # plt.ylabel('$return$')
    # # plt.title(f'return difference of different maps')
    # plt.legend()
    # plt.savefig(f'figs/pdf/return_diff_{test_id}.pdf', dpi=300)
    # plt.savefig(f'figs/png/return_diff_{test_id}.png', dpi=300)

    # plt.figure()
    # plt.plot(prob_list, powers, label='power')
    # plt.xlabel('reward probability $p$')
    # plt.ylabel('power')
    # # plt.title(f'power of different maps')
    # plt.legend()
    # plt.savefig(f'figs/pdf/power_{test_id}.pdf', dpi=300)
    # plt.savefig(f'figs/png/power_{test_id}.png', dpi=300)


if __name__ == "__main__":
    main()
    # self_detect()