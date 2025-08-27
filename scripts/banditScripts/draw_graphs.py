import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from value_detect import value_single_test
from trajctory_detect import traj_single_test

def draw_graphs():

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})

    df = pd.read_csv(f'tables/value_test.csv')
    power_value = df['powers']
    # df = pd.read_csv(f'tables/pedm.csv')
    # power_pedm = df['powers']
    # df = pd.read_csv(f'tables/gmm.csv')
    # power_gmm = df['powers']
    # df = pd.read_csv(f'tables/gaussian.csv')
    # power_gaussian = df['powers']
    # df = pd.read_csv(f'tables/forest.csv')
    # power_forest = df['powers']
    df = pd.read_csv(f'tables/lrt.csv')
    power_lrt = df['powers']
    # df = pd.read_csv(f'tables/reward_test.csv')
    # power_reward = df['powers']

    p_list = np.arange(0.9, 0.1, -0.01)
    plt.figure()
    plt.plot(p_list, power_value, label='value test')
    # plt.plot(p_list, power_pedm, label='pedm')
    # plt.plot(p_list, power_gmm, label='gmm')
    # plt.plot(p_list, power_gaussian, label='gaussian')
    # plt.plot(p_list, power_forest, label='forest')
    plt.plot(p_list, power_lrt, label='lrt')
    # plt.plot(p_list, power_reward, label='reward test')
    # plt.scatter(p_list[0], power_pedm[0], color='red', marker='^', label='pedm default power')
    # plt.scatter(p_list[0], power_value[0], color='red', label='value test default power')
    # plt.scatter(p_list[0], power_lrt[0], color='red', marker='*', label='lrt default power')
    plt.xlabel(f'$p_a$')
    plt.ylabel('Power')
    # plt.xlim(0.8, 0.9)
    plt.legend()
    plt.savefig(f'figs/pdf/power_comp.pdf', dpi=300)
    plt.savefig(f'figs/png/power_comp.png', dpi=300)

def draw_line_graphs():
    x = np.array([2, 5, 10, 20, 40, 50, 80, 100])
    y_value_mean = np.array([0.051, 0.052, 0.055, 0.104, 0.072, 0.077, 0.058, 0.054])
    # y_value_std = np.array([0.009, 0.053, 0.052, 0.099, 0.075, 0.099])

    y_rp_mean = np.array([0.095, 0.080, 0.111, 0.133, 0.099, 0.125, 0.115, 0.125])
    # y_rp_std = np.array([0.059, 0.105, 0.069, 0.075, 0.068, 0.091])

    y_ri_mean = np.array([0.141, 0.111, 0.172, 0.204, 0.148, 0.196, 0.176, 0.194])
    # y_ri_std = np.array([0.178, 0.203, 0.122, 0.139, 0.129, 0.164])

    y_lrt_mean = np.array([0.000, 0.000, 0.000, 0.000, 0.125, 0.060, 0.050, 0.070])
    # y_lrt_std = np.array([0.005, 0.028, 0.031, 0.027, 0.026, 0.033])

    plt.figure()
    plt.plot(x, y_value_mean, marker='p', label='value test')
    # plt.fill_between(x, y_value_mean - y_value_std, y_value_mean + y_value_std, alpha=0.5)
    plt.plot(x, y_rp_mean, marker='o', label='reward pair')
    # plt.fill_between(x, y_rp_mean - y_rp_std, y_rp_mean + y_rp_std, alpha=0.5)
    plt.plot(x, y_ri_mean, marker='*', label='reward independent')
    # plt.fill_between(x, y_ri_mean - y_ri_std, y_ri_mean + y_ri_std, alpha=0.5)
    plt.plot(x, y_lrt_mean, marker='D', label='lrt')
    # plt.fill_between(x, y_lrt_mean - y_lrt_std, y_lrt_mean + y_lrt_std, alpha=0.5)
    plt.axhline(y=0.05, color='red', linestyle='--')
    plt.xlabel('sample size')
    plt.ylabel('power')
    plt.legend()

    plt.savefig(f'figs/pdf/power_test.pdf', dpi=300)
    plt.savefig(f'figs/png/power_test.png', dpi=300)

# def draw_hist(x, y, error=None):

#     cmap = plt.get_cmap('Accent')
#     labels = ['value test', 'lrt', 'reward pair', 'reward independent']
#     plt.bar(x, y, color=cmap.colors[:len(x)], label=labels)
    
#     if error is not None:
#         plt.errorbar(x, y, yerr=error, fmt='none', ecolor='grey', capsize=4)

#     plt.title('powers under the null hypothesis')
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     plt.show()


def main():
    x = np.array(['value test', 'lrt', 'reward pair', 'reward independent'])
    n_tests = 5

    value_powers, reward_pair_powers, reward_ind_powers = value_single_test(n_tests=n_tests)
    traj_powers = traj_single_test(n_tests=n_tests)

    y = np.array([np.mean(value_powers), np.mean(traj_powers), np.mean(reward_pair_powers), np.mean(reward_ind_powers)])
    error = np.array([np.std(value_powers), np.std(traj_powers), np.std(reward_pair_powers), np.std(reward_ind_powers)])

    draw_hist(x, y, error)

def draw_hist():

    df = pd.read_csv(f'tables/value_test.csv')
    return_diff = df['return_diff']

def draw_return():
    df = pd.read_csv(f'tables/value_test.csv')
    return_0_means = df['return_0_means']
    return_1_means = df['return_1_means']
    return_diff_means = df['return_diff_means']
    return_0_stds = df['return_0_stds']
    return_1_stds = df['return_1_stds']
    return_diff_stds = df['return_diff_stds']

    p_list = np.arange(0.9, 0.1, -0.01)

    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.plot(p_list, return_0_means, label='$\hat{V}_r^{(0)}$')
    plt.fill_between(p_list, return_0_means - return_0_stds, return_0_means + return_0_stds, alpha=0.5)
    plt.plot(p_list, return_1_means, label='$\hat{V}_r^{(1)}$')
    plt.fill_between(p_list, return_1_means - return_1_stds, return_1_means + return_1_stds, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('return')
    plt.legend()
    plt.savefig(f'figs/pdf/return.pdf', dpi=300)
    plt.savefig(f'figs/png/return.png', dpi=300)

    plt.figure()
    plt.plot(p_list, return_diff_means, label='$\hat{V}_r^{(0)}$-$\hat{V}_r^{(1)}$')
    plt.fill_between(p_list, return_diff_means - return_diff_stds, return_diff_means + return_diff_stds, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('return difference')
    plt.legend()
    plt.savefig(f'figs/pdf/return_diff.pdf', dpi=300)
    plt.savefig(f'figs/png/return_diff.png', dpi=300)


def draw_lrt():
    df = pd.read_csv(f'tables/lrt.csv')
    likelihood_0_mean = df['likelihood0_means']
    likelihood_0_std = df['likelihood0_stds']
    likelihood_1_mean = df['likelihood1_means']
    likelihood_1_std = df['likelihood1_stds']
    log_ratio_mean = df['log_ratio_means']
    log_ratio_std = df['log_ratio_stds']

    p_list = np.arange(0.9, 0.1, -0.01)

    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.plot(p_list, likelihood_0_mean, label='log-likelihood in MDP$_0$')
    plt.fill_between(p_list, likelihood_0_mean - likelihood_0_std, likelihood_0_mean + likelihood_0_std, alpha=0.5)
    plt.plot(p_list, likelihood_1_mean, label='log-likelihood in MDP$_1$')
    plt.fill_between(p_list, likelihood_1_mean - likelihood_1_std, likelihood_1_mean + likelihood_1_std, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('log-likelihood')
    plt.legend()
    plt.savefig(f'figs/pdf/likelihood.pdf', dpi=300)
    plt.savefig(f'figs/png/likelihood.png', dpi=300)

    plt.figure()
    plt.plot(p_list, log_ratio_mean, label='log-likelihood ratio')
    plt.fill_between(p_list, log_ratio_mean - log_ratio_std, log_ratio_mean + log_ratio_std, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('log-likelihood ratio')
    plt.legend()
    plt.savefig(f'figs/pdf/ratio.pdf', dpi=300)
    plt.savefig(f'figs/png/ratio.png', dpi=300)


if __name__ == '__main__':
    # main()
    draw_lrt()
    # draw_return()
    # x = np.array(range(4))
    # y = np.array([0.6, 0.7, 0.8, 0.9])
    # error = np.array([0.1, 0.1, 0.1, 0.1])

    # draw_hist(x, y, error)

    # draw_line_graphs()

    # draw_graphs()