import sys
sys.path.append('../..')
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import *
from envs.gym_randomgrid.mdps.grid_mdp import RandomGridMDP
from envs.gym_randomgrid.generator import MAP_LIB, genEnv, genMDP


# @hydra.main(config_path="../../configs", config_name="gridenv")
# def main(cfg:DictConfig):
#     np.random.seed(1)
#     powers = []
#     diffs = []
#     for map_id in range(cfg.detect.n_maps):
#         default_values = []
#         detect_values = []
#         for _ in range(cfg.detect.n_trials):
#             # dp for default env
#             policy_path = f'save/model/policy/map_{cfg.detect.policy_id}_policy_2000.csv'
#             policy = read_csv(policy_path)[:, -1]
#             mdp = SimpleGridMDP(obstacle_map=MAP_LIB[0], deterministic=False, gamma=0.9)
#             value, _, _ = dp_eval(policy, mdp, discount=0.9, theta=0.00001)
#             default_values.append(value)

#             # mc for detect env
#             env = genEnv(map_id=map_id, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
#             detect_value, _, _ = mc_eval(policy, env, iterations=1000)
#             detect_values.append(detect_value)

    #     power, d = tt_test(default_values, detect_values)
    #     print(f'map {map_id}: {power}, {d}')
    #     powers.append(power)
    #     diffs.append(d)
    # # plt.plot(range(cfg.detect.n_maps), powers)
    # # plt.xlabel('map id')
    # # plt.ylabel('power')
    # # plt.annotate('default', (0, powers[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # # plt.title(f'power of different maps')
    # # plt.savefig(f'save/detect/value detection.png')
        
    # plt.plot(range(cfg.detect.n_maps), diffs)
    # plt.xlabel('map id')
    # plt.ylabel(' Cohen\'s d')
    # plt.annotate('default', (0, powers[0]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.title(f' Cohen\'s d of different maps')
    # plt.savefig(f'save/detect/value  Cohen\'s d.png')

@hydra.main(config_path="../../configs", config_name="gridenv")
def main(cfg:DictConfig):
    np.random.seed(1)
    policy_path = f'save/model/policy/map_{cfg.detect.policy_id}_policy_1000.csv'
    policy = read_csv(policy_path)[:, -1]
    mean_values = []
    std_values = []
    
    for i in range(10):

        env = genEnv(map_id=i, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
        returns = compute_returns(policy, env, iterations=cfg.detect.collect_trajs)
        mean_values.append(np.mean(returns))
        std_values.append(np.std(returns))

    x_arr = np.array(range(10))
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)

    plt.figure()
    plt.plot(x_arr, mean_values)
    # plt.annotate('default', (0, mean_values[0]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.fill_between(x_arr, mean_values-std_values, mean_values+std_values, alpha=0.2)

    plt.xlabel('map id')
    plt.ylabel('returns')
    plt.title('returns of different maps')
    save_dir = os.path.join(cfg.detect.save_dir, f'returns.png')
    plt.savefig(save_dir)

if __name__ == "__main__":
    main()
    # self_detect()