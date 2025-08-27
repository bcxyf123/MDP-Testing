import os
import sys
sys.path.append('../..')
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import seaborn as sns
from gymEnvGen import *
from gym_detect import *

def smooth(data, sm=5):
    d = data
    z = np.ones(len(d))
    y = np.ones(sm)*1.0
    d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
    
    return d


@hydra.main(config_path='../../configs', config_name='config')
def draw_graphs(cfg: DictConfig):

    env_name = 'MPE_reward'
    plt.rcParams.update({'font.size': 32})
    plt.rcParams['font.family'] = 'Times New Roman'
    
    color_palette = sns.color_palette("tab10")  
    
    # param_name = 'LINK_LENGTH_1'
    # print(param_name)
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/value_new.csv')
    # power_value = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/pedm.csv')
    # power_pedm = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/riqn.csv')
    # power_riqn = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/gmm.csv')
    # power_gmm = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/gaussian.csv')
    # power_gaussian = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/knn.csv')
    # power_knn = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/forest.csv')
    # power_forest = df['powers']
    
    df = pd.read_csv(f'tables/{env_name}/value_new.csv')
    power_value = df['powers']
    df = pd.read_csv(f'tables/{env_name}/pedm.csv')
    power_pedm = df['powers']
    df = pd.read_csv(f'tables/{env_name}/riqn.csv')
    power_riqn = df['powers']
    df = pd.read_csv(f'tables/{env_name}/gmm.csv')
    power_gmm = df['powers']
    df = pd.read_csv(f'tables/{env_name}/gaussian.csv')
    power_gaussian = df['powers']
    df = pd.read_csv(f'tables/{env_name}/knn.csv')
    power_knn = df['powers']
    df = pd.read_csv(f'tables/{env_name}/forest.csv')
    power_forest = df['powers']
    # df = pd.read_csv(f'tables/{env_name}/{param_name}/lrt.csv')
    # power_lrt = df['powers']

    # env, contexts, default_id = genEnv(env_name=env_name, param_name=param_name)
    # p_list = []
    # for c in contexts:
    #     if c:
    #         p_list.append(contexts[c][param_name])
    p_list = np.arange(0.1, 10.1, 0.1)

    power_value = smooth(power_value, sm=3)
    power_pedm = smooth(power_pedm, sm=3)
    power_riqn = smooth(power_riqn, sm=3)
    power_gmm = smooth(power_gmm, sm=3)
    power_gaussian = smooth(power_gaussian, sm=3)
    power_knn = smooth(power_knn, sm=3)
    power_forest = smooth(power_forest, sm=3)
    # power_lrt = smooth(power_lrt)
            
    p_list = p_list

    plt.figure(figsize=(10, 8))
    plt.plot(p_list, power_value, color=color_palette[0])
    plt.plot(p_list, power_pedm, color=color_palette[2])
    plt.plot(p_list, power_riqn, label='RIQN', color=color_palette[3])
    plt.plot(p_list, power_gmm, label='GMM', color=color_palette[4])
    plt.plot(p_list, power_gaussian, color=color_palette[5])
    plt.plot(p_list, power_knn, label='KNN', color=color_palette[6])
    # sns.lineplot(x=p_list, y=power_forest, label='forest')

    plt.xlabel('Arena Size')
    plt.ylabel('Power')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    # plt.tight_layout()
    # plt.legend(loc='lower right')
    # plt.legend()

    # save_dir = f'figs/{env_name}/power_{param_name}'
    save_dir = f'figs/{env_name}/power'

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    plt.savefig(f'{save_dir}.pdf', dpi=300)
    plt.savefig(f'{save_dir}.png', dpi=300)


def draw_tensorboard():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    plt.rcParams['font.size'] = 12
    # 路径到你的TensorBoard日志文件
    log_path = 'models/ppo_acrobot/PPO_3'

    # 创建一个EventAccumulator实例
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()  # 加载日志数据

    # 假设我们要提取标量 'loss'，这里你需要替换为你自己的标量名称
    # 获取所有的loss数据
    loss_values = event_acc.Scalars('rollout/ep_len_mean')

    # 提取step和loss值
    steps = [item.step for item in loss_values]
    losses = [item.value for item in loss_values]

    # 使用matplotlib绘图
    plt.plot(steps, losses)
    plt.xlabel('Steps')
    plt.ylabel('Episode Length')
    # plt.show()
    plt.savefig('figs/ppo_acrobot_length.pdf', dpi=300)


def draw_returns():

    param_names = 'MAX_VEL_2'

    df = pd.read_csv(f'tables/Acrobot/{param_names}/value_test.csv')
    return_0_means = df['return_0_means']
    return_0_stds = df['return_0_stds']
    return_1_means = df['return_1_means']
    return_1_stds = df['return_1_stds']
    return_diff_means = df['return_diff_means']
    return_diff_stds = df['return_diff_stds']

    plt.rcParams['font.size'] = 12
    plt.figure()
    plt.plot(return_0_means, label='$\hat{V}(0)$')
    plt.fill_between(np.arange(len(return_0_means)), return_0_means - return_0_stds, return_0_means + return_0_stds, alpha=0.2)
    plt.plot(return_1_means, label='$\hat{V}(1)$')
    plt.fill_between(np.arange(len(return_1_means)), return_1_means - return_1_stds, return_1_means + return_1_stds, alpha=0.2)
    plt.xlabel(f'{param_names}')
    plt.ylabel('Return')
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig(f'figs/acrobot/{param_names}/return.pdf', dpi=300)

    plt.figure()
    plt.plot(return_diff_means, label='$\hat{V}(0)-\hat{V}(1)$')
    plt.fill_between(np.arange(len(return_diff_means)), return_diff_means - return_diff_stds, return_diff_means + return_diff_stds, alpha=0.2)
    plt.xlabel(f'{param_names}')
    plt.ylabel('Return')
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig(f'figs/acrobot/{param_names}/return_diff.pdf', dpi=300)


if __name__ == '__main__':
    # draw_returns()
    draw_graphs()