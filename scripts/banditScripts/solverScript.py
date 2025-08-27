import sys
sys.path.append('../..')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import beta

from utils.utils import *
from envs.bandit.bandit import BernoulliBandit

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        # self.reset()

    def reset(self):
        pass

    def update_regret(self, k, reward):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        self.rewards.append(reward)
        # self.regret += self.bandit.best_rew - reward
        # self.regrets.append(self.regret)
    
    # def normalize_regret(self):
    #     self.regret = self.regret/len(self.regrets)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps, update=True):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k, r = self.run_one_step(update)
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k, r)
        # self.reset()

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super().__init__(bandit)
        self.policy_fixed = False
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.
        self.actions = []
        self.regrets = []
        self.rewards = []
        if not self.policy_fixed:
            self._a = np.ones(self.bandit.K)
            self._b = np.ones(self.bandit.K)
        self._a_bar = np.ones(self.bandit.K)
        self._b_bar = np.ones(self.bandit.K)

    def run_one_step(self, update=True):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        # k = self.choose_action()
        r = self.bandit.step(k)
        if update:
            self.update_beta_pdf(k, r)
        self.update_bernoulli_pdf(k, r)
        return k, r
    
    def choose_action(self, ):
        # Normalize self._a_bar to get probabilities
        probabilities = self._a_bar / self._a_bar.sum()
        # Choose an action based on the probabilities
        k = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        return k

    def update_beta_pdf(self, k, r):
        self._a[k] += r
        self._b[k] += (1 - r)
    
    def update_bernoulli_pdf(self, k, r):
        self._a_bar[k] += r
        self._b_bar[k] += (1 - r)
    
    def calculate_bernoulli_pdf(self, k, r):
        prob = (self._a_bar[k]-1)/(self._a_bar[k] + self._b_bar[k]-2) if r == 1 else 1-(self._a_bar[k]-1)/(self._a_bar[k] + self._b_bar[k]-2)
        return prob

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

    

def test_env_params():
    K = 5
    default_p = 0.1
    n_trials = 50
    test_p_list = np.arange(0.1, 1.0, 0.01)

    powers = []
    diff_mean = []
    diff_std = []

    default_bandit = BernoulliBandit(p=default_p)
    default_solver = ThompsonSampling(default_bandit)

    for test_p in test_p_list:
        print(f'test_p: {test_p}')
        default_returns = []
        test_returns = []
        diff = []

        test_bandit = BernoulliBandit(p=test_p)
        test_solver = ThompsonSampling(test_bandit)

        for _ in range(n_trials):
            default_solver.run(10000)
            default_returns.append(np.mean(default_solver.rewards))
            test_solver.run(10000)
            test_returns.append(np.mean(test_solver.rewards))
            diff.append(np.mean(test_solver.rewards) - np.mean(default_solver.rewards))
            default_solver.reset()
            test_solver.reset()

        power = t_paired_test(default_returns, test_returns)
        powers.append(power)

        diff_mean.append(np.mean(diff))
        diff_std.append(np.std(diff))

        print()

    # plot power
    # total_power_arr = np.array(total_powers)
    # data = np.vstack((np.array(test_p_list), total_power_arr))
    # arr = data
    # x_arr = arr[0, :]
    # power_arr = arr[1:, :]
    # df = pd.DataFrame(power_arr).T

    # mean_values = df.mean(axis=1)
    # std_values = df.std(axis=1)

    plt.figure()
    plt.plot(test_p_list, powers)
    # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)
    plt.annotate('default', (0.1, powers[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('arm probability')
    plt.ylabel('power')
    plt.title(f'power of return')
    plt.savefig(f'env_params_test_power.png')

    # # plot regret diff
    # regret_diff_means = [np.mean(diff) for diff in regret_diff]
    # regret_diff_stds = [np.std(diff) for diff in regret_diff]
    diff_mean = np.array(diff_mean)
    diff_std = np.array(diff_std)
    plt.figure()
    plt.plot(np.array(test_p_list), diff_mean)
    plt.fill_between(test_p_list, diff_mean-diff_std, diff_mean+diff_std, alpha=0.2)
    plt.annotate('default', (0.1, diff_mean[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('arm probability')
    plt.ylabel('return difference')
    plt.title(f'return different')
    plt.savefig(f'env_params_test_return.png')

def test_sample_size():
    K = 5
    default_p = 0.1
    n_trials = 100
    test_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性\
    returns0 = []
    for _ in range(20):
        default_bandit = BernoulliBandit(p=default_p)
        thompson_sampling_solver = ThompsonSampling(default_bandit)
        thompson_sampling_solver.run(5000)
        rewards = np.mean(thompson_sampling_solver.rewards)
        returns0.append(rewards)

    powers = []
    mean_values = []
    std_values = []
    for test_p in test_p_list:
        returns1 = []
        for _ in range(20):            
            bandit = BernoulliBandit(p=test_p)
            thompson_sampling_solver = ThompsonSampling(bandit)
            thompson_sampling_solver.run(5000)

            rewards = np.sum(thompson_sampling_solver.rewards)
            returns1.append(rewards)

        mean_values.append(np.mean(returns1))
        std_values.append(np.std(returns1))
        
            # power = tt_test(default_regrets, test_regrets)
            # powers.append(power)
            # print(f'{test_p} {n_trials} power: {power}')
            # regret_diff.append(np.mean(regret_diff_mean))
            # print(f'{test_p} {n_trials} regret: {np.mean(regret_diff_mean)}')

        power = t_ind_test(returns0, returns1)
        powers.append(power)
        print(power)

    mean_values = np.array(mean_values)
    std_values = np.array(std_values)

    print(mean_values, std_values)

    plt.figure()
    plt.plot(np.array(test_p_list), mean_values)
    plt.fill_between(test_p_list, mean_values-std_values, mean_values+std_values, alpha=0.2)
    plt.annotate('default', (0.1, mean_values[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('arm probability')
    plt.ylabel('return')
    plt.title(f'returns of different arm probability')
    plt.savefig(f'returns.png')

    plt.figure()
    plt.plot(test_p_list, powers)
    # plt.fill_between(test_p_list, mean_values-std_values, mean_values+std_values, alpha=0.2)
    plt.annotate('default', (0.1, powers[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('arm probability')
    plt.ylabel('power')
    plt.title(f'power of return')
    plt.savefig(f'return power.png')
                
        # total_powers.append(powers)
        # total_regret_diff.append(regret_diff)

        # total_power_arr = np.array(total_powers)
        # data = np.vstack((np.array(sample_sizes), total_power_arr))
        # arr = data
        # x_arr = arr[0, :]
        # power_arr = arr[1:, :]
        # df = pd.DataFrame(power_arr).T

        # mean_values = df.mean(axis=1)
        # std_values = df.std(axis=1)

        # plt.figure()
        # plt.plot(x_arr, mean_values)
        # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)
        # plt.xlabel('sample size')
        # plt.ylabel('power')
        # plt.title(f'default arm probability {test_p}')
        # plt.savefig(f'sample_size_power_{test_p}.png')

        # # plot regret diff
        # total_regret_arr = np.array(total_regret_diff)
        # data = np.vstack((np.array(sample_sizes), total_regret_arr))
        # arr = data
        # x_arr = arr[0, :]
        # regret_arr = arr[1:, :]
        # df = pd.DataFrame(regret_arr).T

        # mean_values = df.mean(axis=1)
        # std_values = df.std(axis=1)

        # plt.figure()
        # plt.plot(x_arr, mean_values)
        # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)
        # plt.xlabel('sample size')
        # plt.ylabel('regret difference')
        # plt.title(f'default arm probability {test_p}')
        # plt.savefig(f'sample_size_regret_{test_p}.png')

def self_test():
    K = 5
    default_p = 0.1
    test_p = 0.1
    n_trials = 50

    np.random.seed(222333)  # 设定随机种子,使实验具有可重复性

    default_regrets = []
    test_regrets = []
    powers = []
    regret_diffs = []
    for _ in range(10):
        regret_diff_mean = []
        for _ in range(n_trials):
            bandit = BernoulliBandit(p=default_p)
            thompson_sampling_solver = ThompsonSampling(bandit)
            thompson_sampling_solver.run(10000)
            default_regret = thompson_sampling_solver.regret
            default_regrets.append(default_regret)
            
            bandit = BernoulliBandit(p=test_p)
            thompson_sampling_solver = ThompsonSampling(bandit)
            thompson_sampling_solver.run(10000)
            test_regret = thompson_sampling_solver.regret
            test_regrets.append(test_regret)

            regret_diff_mean.append(test_regret - default_regret)

        power = TTestIndPower(default_regrets, test_regrets)
        # print(f'power: {power}')
        powers.append(power)
        regret_diffs.append(np.mean(regret_diff_mean))

    print(f'mean: {np.mean(powers)}, std: {np.std(powers)}')
    print(f'mean: {np.mean(regret_diffs)}, std: {np.std(regret_diffs)}')


def trajectory_test(default_p=0.1, test_p=0.1):

    print(f'{test_p}')
    default_bandit = BernoulliBandit(p=default_p)

    test_bandit = BernoulliBandit(p=test_p)
    test_solver = ThompsonSampling(test_bandit)

    n_trials = 100
    ratio_list = []
    for i in range(n_trials):
        test_solver.run(10000)
        actions = test_solver.actions
        rewards = test_solver.rewards
        likelihood0 = (sum(np.log(default_bandit.probs[a] if r == 1 else 1 - default_bandit.probs[a]) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        likelihood1 = (sum(np.log(test_solver.calculate_bernoulli_pdf(a, r)) 
                           for a, r in zip(actions, rewards) if r in [0, 1]))
        ratio = -2*(likelihood0 - likelihood1)
        ratio_list.append(np.log(ratio))
        test_solver.reset()

    print(f'{np.mean(ratio_list):.3f}', f'{np.std(ratio_list):.3f}')
    # power = chi_test(ratio_list, df=5)

    # # plot histogram
    # plt.figure()
    # plt.hist(ratio_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True)
    # plt.title('likelihood ratio distribution')
    # plt.xlabel('likelihood ratio')
    # plt.ylabel('frequency')
    # plt.savefig(f'ratio_{n_trials}.png', dpi=300)

    return np.array(ratio_list)


def main():

    default_p = 0.6
    test_p = 0.6
    n_trials = 50

    default_bandit = BernoulliBandit(p=default_p)
    default_solver = ThompsonSampling(default_bandit)

    test_bandit = BernoulliBandit(p=test_p)
    test_solver = ThompsonSampling(test_bandit) 

    return_diff = []
    power_list = []

    # for _ in range(10):
    returns1 = []
    returns2 = []
    for _ in range(n_trials):
        default_solver.run(10000)
        test_solver.run(10000)
        returns1.append(np.mean(default_solver.rewards))
        returns2.append(np.mean(test_solver.rewards))
        return_diff.append(np.mean(test_solver.rewards) - np.mean(default_solver.rewards))
        default_solver.reset()
        test_solver.reset()

    # plot histogram
    plt.figure()
    plt.hist(returns1, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True)
    plt.title('return distribution')
    plt.xlabel('return value')
    plt.ylabel('frequency')
    plt.savefig(f'return_{test_p}_{n_trials}.png', dpi=300)

    # plot histogram
    plt.figure()
    plt.hist(return_diff, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True)
    plt.title('return distribution')
    plt.xlabel('return difference value')
    plt.ylabel('frequency')
    plt.savefig(f'return_difference_{test_p}_{n_trials}.png', dpi=300)
        # print(returns1, returns2)
    print(f'{np.mean(returns1):.3f}', f'{np.std(returns1):.3f}')
    print(f'{np.mean(return_diff):.2e}', f'{np.std(return_diff):.2e}')

    power = t_paired_test(returns1, returns2)
    # power_list.append(power)
    # power_list.append(power)
    # print(f'{np.mean(return_diff):.2e}', f'{np.std(return_diff):.2e}')
    # print(power_list)
    # print(f'{np.mean(power_list):.3f}', f'{np.std(power_list):.3f}')

    # # plot histogram
    # plt.figure()
    # plt.hist(power_list, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True)
    # plt.title('power distribution')
    # plt.xlabel('power')
    # plt.ylabel('frequency')
    # plt.savefig(f'power_{n_trials}.png', dpi=300)

    return power, return_diff

if __name__ == "__main__":

    # test_sample_size()
    test_env_params()
    # self_test()
    
    # main()

    # trajectory_test(default_p=0.1, test_p=0.1)

    # p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # ratio_p_list = []
    # for p in p_list:
    #     ratio_p_list.append(trajectory_test(default_p=0.1, test_p=p))
    # ratio_p_arr = np.array(ratio_p_list).T
    
    # x_arr = np.array(p_list)
    # ratio_arr = np.array(ratio_p_arr)
    # df = pd.DataFrame(ratio_arr).T

    # mean_values = df.mean(axis=1)
    # std_values = df.std(axis=1)

    # plt.plot(x_arr, mean_values)
    # plt.annotate('default', (0.1, mean_values[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)

    # plt.xlabel('arm probability')
    # plt.ylabel('log likelihood Ratio')
    # plt.title('log likelihood ratio of different arm probability')
    # plt.savefig(f'likelihood_test.png')