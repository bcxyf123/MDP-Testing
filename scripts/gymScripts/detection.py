import os
import sys
sys.path.append('../..')
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils import *
from stable_baselines3 import PPO
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from scripts.gymScripts.gymEnvGen import generate_AcrobotEnvs, generate_CartPoleEnvs
from train_transition_model import *

def detection(env, policy, model0, model1, num_trajs):
    # collect trajectories in the detected env
    trajs = collect_trajectories(env, policy, num_trajs)
    # # detect the env
    # likelihood_0_list = []
    # likelihood_1_list = []
    # for trj in trajs:
    #     likelihood_0 = compute_likelihood(trj, model0)
    #     likelihood_0_list.append(likelihood_0)
    #     likelihood_1 = compute_likelihood(trj, model1)
    #     likelihood_1_list.append(likelihood_1)
    # likelihood_0 = np.mean(likelihood_0_list)
    # likelihood_1 = np.mean(likelihood_1_list)
    ratio_list = []
    for trj in trajs:
        _, _, ratio = likelihoodRatio(trj, model0, model1)
        ratio_list.append(ratio)
    ratio = np.mean(ratio_list)
    power = chi_test(ratio_list, df=2)
    # print("ratio list: ", ratio_list)
    # print("Likelihood ratio: ", ratio)

    return ratio, power

@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    # # generate env
    # envcfg = cfg.env.Acrobot
    # env, contexts = generate_AcrobotEnvs(param)
    # total_ratio_list = []
    # seeds = [random.randint(0, 1000000) for _ in range(10)]
    # for i in range(10):
    #     ratio_list = []
    #     x_list = []
    #     set_random_seed(seeds[i])
    #     print("seed: ", seeds[i])
    #     # device = torch.device(cfg.train.cuda if torch.cuda.is_available() else "cpu")
    #     for j in range(1, len(contexts)):
    #         # j = 6
    #         env.context_id = j
    #         # env.context_id = cfg.detect.env_id
    #         print(f"Currently using {param_name}: ", env.context[param_name])
    #         x_list.append(env.context[param_name])
    #         # load policy
    #         policy = PPO.load(cfg.train.load_dir)
    #         # load models
    #         model0 = TransitionModel(envcfg.observation_dim, envcfg.action_dim).to(device)
    #         model0.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models', 
    #                                                         f'transition_model_0_16.pth')))
    #         model1 = TransitionModel(envcfg.observation_dim, envcfg.action_dim).to(device)
    #         model1.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models', 
    #                                                         param, f'transition_model_{j}.pth')))
    #         # detection
    #         ratio = detection(env, policy, model0, model1, cfg.detect.collect_trajs)
    #         print("Likelihood ratio: ", ratio)
    #         ratio_list.append(ratio)
    #     # total_ratio_list.append(ratio)
    # # print(np.mean(total_ratio_list), np.std(total_ratio_list))
    #     total_ratio_list.append(ratio_list)
    # x_arr = np.array(x_list)
    # total_ratio_arr = np.array(total_ratio_list)
    # data = np.vstack((x_arr, total_ratio_arr))
    # df = pd.DataFrame(data)
    # df.to_csv(f'models/acrobot_models/ratio_{param_name}.csv', index=False, header=False)

    # old-new detection
    # generate env
    # np.random.seed(1)

    env_name = 'Acrobot'
    param_name = 'LINK_LENGTH_1'
    envcfg = cfg.env.Acrobot

    param_name_list, policy_name = Parameter_name_list(env_name)
    policy = PPO.load(os.path.join(cfg.train.load_dir, policy_name))

    for param_name in param_name_list:

        env, contexts, _ = genEnv(env_name, param_name)

        ratio_list = []
        x_list = []
        power_list = []

        for j in range(1, len(contexts)):
            env.context_id = j
            env.reset()
            # env.context_id = cfg.detect.env_id
            print(f"Currently using {param_name}: ", env.context[param_name])
            x_list.append(env.context[param_name])
            # load models
            model0 = TransitionModel(envcfg.observation_dim, action_dim=1).to(device)
            model0.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models',
                                                            f'transition_model_0.pth')))
            model1 = TransitionModel(envcfg.observation_dim, action_dim=1).to(device)
            model1.load_state_dict(torch.load(os.path.join(cfg.train.save_dir, 'acrobot_models', f'{param_name}',
                                                            f'transition_model_{j}.pth')))
            # detection
            ratio, power = detection(env, policy, model0, model1, cfg.detect.collect_trajs)
            print("Likelihood ratio: ", ratio)
            print("power: ", power)
            print()
            ratio_list.append(ratio)
            power_list.append(power)

        data = {
            'ratio': ratio_list,
            'power': power_list
        }

        df = pd.DataFrame(data)
        df.to_csv(f'tables/{env_name}/{param_name}/lrt.csv', index=False, header=False)

    # plt.plot(x_list, ratio_list)
    # plt.annotate('default', (0.0, ratio_list[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)

    # plt.xlabel(param)
    # plt.ylabel('likelihood ratio')
    # plt.title(f'likelihood ratio of different {param_name}')
    # plt.savefig(f'models/acrobot_models/ratio_{param_name}.png')
        
        # plt.hist(ratio_list, bins='auto')
        # plt.show()

        # total_ratio_list.append(ratio_list)
    # x_arr = np.array(x_list)
    # total_ratio_arr = np.array(total_ratio_list)
    # data = np.vstack((x_arr, total_ratio_arr))
    # df = pd.DataFrame(data)
    # df.to_csv(f'models/acrobot_models/ratio_{param_name}.csv', index=False, header=False)

    # # 创建一个图形
    # plt.figure()

    # # 绘制列表中的值
    # plt.plot(ratio_list, marker='o')  # 使用圆圈标记每个点

    # # 在第三个元素上添加注释
    # # 注意：列表索引从0开始，所以第三个元素的索引是2
    # plt.annotate('default', (9, ratio_list[9]), textcoords="offset points", xytext=(0,10), ha='center')

    # # 显示图形
    # plt.savefig('ratio.png')

    # arr = np.array(pd.read_csv(f'models/acrobot_models/ratio_{param_name}.csv', header=None))
    # x_arr = arr[0, :]
    # ratio_arr = arr[1:, :]
    # df = pd.DataFrame(ratio_arr).T

    # mean_values = df.mean(axis=1)
    # std_values = df.std(axis=1)

    # plt.plot(x_arr, mean_values)
    # plt.annotate('default', (0.0, mean_values[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)

    # plt.xlabel(param)
    # plt.ylabel('Likelihood Ratio')
    # # plt.legend()
    # plt.savefig(f'models/acrobot_models/ratio_{param_name}.png')



if __name__ == '__main__':
    main()

    # # plot
    # for i in range(ratio_arr.shape[0]):
    #     sns.lineplot(x=x_arr, y=ratio_arr[i], errorbar='sd')
    # plt.savefig('ratio.png')