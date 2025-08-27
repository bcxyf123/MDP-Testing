import sys
sys.path.append('../..')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from utils.utils import *

def draw_graphs():

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['font.family'] = 'Times New Roman'
    
    color_palette = sns.color_palette("tab10")  # 获取默认的颜色循环

    # df = pd.read_csv(f'data/return_prob_test_results.csv')
    # power_value = df['powers']
    # df = pd.read_csv(f'data/lrt_prob_test_results.csv')
    # power_lrt = df['powers']
    # df = pd.read_csv(f'tables/prob_pedm.csv')
    # power_pedm = df['powers']
    # df = pd.read_csv(f'tables/prob_riqn.csv')
    # power_riqn = df['powers']
    # df = pd.read_csv(f'tables/prob_gmm.csv')
    # power_gmm = df['powers']
    # df = pd.read_csv(f'tables/prob_gaussian.csv')
    # power_gaussian = df['powers']
    # df = pd.read_csv(f'tables/prob_knn.csv')
    # power_knn = df['powers']

    df = pd.read_csv(f'data/return_map_test_results.csv')
    power_value = df['powers']
    df = pd.read_csv(f'tables/map_lrt.csv')
    power_lrt = df['powers']
    df = pd.read_csv(f'tables/map_pedm.csv')
    power_pedm = df['powers']
    df = pd.read_csv(f'tables/map_riqn.csv')
    power_riqn = df['powers']
    df = pd.read_csv(f'tables/map_gmm.csv')
    power_gmm = df['powers']
    df = pd.read_csv(f'tables/map_gaussian.csv')
    power_gaussian = df['powers']
    df = pd.read_csv(f'tables/map_knn.csv')
    power_knn = df['powers']


    # power_value = smooth(power_value, sm=3)
    # power_lrt = smooth(power_lrt, sm=3)
    # power_pedm = smooth(power_pedm, sm=3)
    # power_riqn = smooth(power_riqn, sm=3)
    # power_gmm = smooth(power_gmm, sm=3)
    # power_gaussian = smooth(power_gaussian, sm=3)
    # power_knn = smooth(power_knn, sm=3)
    # # power_forest = smooth(power_forest, sm=3)

    p_list = np.arange(0.9, 0.1, -0.01)
    map_list = range(7)

    plt.figure(figsize=(10, 8))

    plt.plot(map_list, power_value, marker='*', color=color_palette[0])
    plt.plot(map_list, power_lrt, marker='o', label='trajectory test (ours)', color=color_palette[1])
    plt.plot(map_list, power_pedm, marker='^', color=color_palette[2])
    plt.plot(map_list, power_riqn, marker='s', label='RIQN', color=color_palette[3])
    plt.plot(map_list, power_gmm, marker='.', color=color_palette[4])
    plt.plot(map_list, power_gaussian, marker='v', color=color_palette[5])
    plt.plot(map_list, power_knn, marker='p', color=color_palette[6])
    # plt.plot(map_list, power_forest, marker='x', label='forest')

    # plt.plot(p_list, power_value, label='value test')
    # plt.plot(p_list, power_lrt, label='trajectory test' )
    # plt.plot(p_list, power_pedm)
    # plt.plot(p_list, power_riqn, )
    # plt.plot(p_list, power_gmm, )
    # plt.plot(p_list, power_gaussian,)
    # plt.plot(p_list, power_knn, )
    # # plt.plot(p_list, power_forest, label='forest')

    plt.xlabel(f'Map ID')
    plt.ylabel('Power')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
    # plt.tight_layout()
    plt.grid()
    plt.savefig(f'figs/pdf/power_comp_map.pdf', dpi=300)

    # plt.xlabel('$p_t$')
    # plt.ylabel('Power')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    # plt.tight_layout()
    # plt.savefig(f'figs/pdf/power_comp_prob.pdf', dpi=300)


def draw_return():
    df = pd.read_csv(f'tables/map_return.csv')
    return_0_means = df['return_0_means']
    return_1_means = df['return_1_means']
    return_diff_means = df['return_diff_means']
    return_0_stds = df['return_0_stds']
    return_1_stds = df['return_1_stds']
    return_diff_stds = df['return_diff_stds']

    p_list = np.arange(0.9, 0.1, -0.01)
    p_list = range(7)

    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.plot(p_list, return_0_means, label='$\hat{V}_t^{(0)}$')
    plt.fill_between(p_list, return_0_means - return_0_stds, return_0_means + return_0_stds, alpha=0.5)
    plt.plot(p_list, return_1_means, label='$\hat{V}_t^{(1)}$')
    plt.fill_between(p_list, return_1_means - return_1_stds, return_1_means + return_1_stds, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('return')
    plt.legend()
    plt.savefig(f'figs/pdf/return_map.pdf', dpi=300)
    plt.savefig(f'figs/png/return_map.png', dpi=300)

    plt.figure()
    plt.plot(p_list, return_diff_means, label='$\hat{V}_t^{(0)}$-$\hat{V}_t^{(1)}$')
    plt.fill_between(p_list, return_diff_means - return_diff_stds, return_diff_means + return_diff_stds, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('return difference')
    plt.legend()
    plt.savefig(f'figs/pdf/return_diff_map.pdf', dpi=300)
    plt.savefig(f'figs/png/return_diff_map.png', dpi=300)


def draw_lrt():
    df = pd.read_csv(f'tables/map_lrt.csv')
    likelihood_0_mean = df['likelihood_0_means']
    likelihood_0_std = df['likelihood_0_stds']
    likelihood_1_mean = df['likelihood_1_means']
    likelihood_1_std = df['likelihood_1_stds']
    ratio_mean = df['ratio_means']
    ratio_std = df['ratio_stds']

    p_list = np.arange(0.9, 0.1, -0.01)
    p_list = range(7)

    plt.rcParams.update({'font.size': 12})
    plt.figure()
    plt.plot(p_list, likelihood_0_mean, label='log-likelihood in MDP$_0$')
    plt.fill_between(p_list, likelihood_0_mean - likelihood_0_std, likelihood_0_mean + likelihood_0_std, alpha=0.5)
    plt.plot(p_list, likelihood_1_mean, label='log-likelihood in MDP$_1$')
    plt.fill_between(p_list, likelihood_1_mean - likelihood_1_std, likelihood_1_mean + likelihood_1_std, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('log-likelihood')
    plt.legend()
    plt.savefig(f'figs/pdf/likelihood_map.pdf', dpi=300)
    plt.savefig(f'figs/png/likelihood_map.png', dpi=300)

    plt.figure()
    plt.plot(p_list, ratio_mean, label='log-likelihood ratio')
    plt.fill_between(p_list, ratio_mean - ratio_std, ratio_mean + ratio_std, alpha=0.5)
    plt.xlabel('$p_a$')
    plt.ylabel('log-likelihood ratio')
    plt.legend()
    plt.savefig(f'figs/pdf/ratio_map.pdf', dpi=300)
    plt.savefig(f'figs/png/ratio_map.png', dpi=300)


if __name__ == '__main__':
    draw_graphs()
    # draw_return()
    # draw_lrt()