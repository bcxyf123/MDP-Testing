import sys
sys.path.append('../..')
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import *
from envs.gym_simplegrid.mdps.grid_mdp import SimpleGridMDP
from envs.gym_simplegrid.generator import MAP_LIB, genEnv, genMDP
from algo.dp_eval import dp_eval
from algo.value_eval import mc_eval
from testers.base import *

from scipy.stats import gaussian_kde


class maze_return_tester(t_tester):
    def __init__(self, is_map_test, policy, n_trials=10, gamma=0.99):
        super().__init__(
            n_trials=n_trials,
        )
        self.is_map_test = is_map_test
        self.policy = policy
        self.gamma = gamma
        self.default_map = 0
        self.default_prob = 0.9

    def sample(
            self,
            test_map: int = 0,
            test_prob: float = 0.9,
            is_save_metadata: bool = False,
        ) -> np.ndarray:
        
        # fix prob in map test
        if self.is_map_test:
            test_prob = self.default_prob
            save_path = f'data/return_test_metadata_map_{test_map}.csv'
        # fix map in prob test
        else:
            test_map = self.default_map
            save_path = f'data/return_test_metadata_prob_{test_prob}.csv'

        default_env = genEnv(map_size=(4,8), map_id=self.default_map, max_prob=self.default_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
        test_env = genEnv(map_size=(4,8), map_id=test_map, max_prob=test_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

        # compute returns 0
        returns_0 = compute_returns(self.policy, default_env, gamma=self.gamma, iterations=self.n_trials)
        # compute returns 1
        returns_1 = []
        for _ in range(self.n_trials):
            return_1 = 0
            state, _ = test_env.reset()
            default_env.reset()
            step = 0
            done = False
            while not done:
                action = self.policy[state]
                next_state, _, done, _, info = test_env.step(action)
                x, y = info['x'], info['y']
                reward = default_env.unwrapped.get_reward(x, y)
                return_1 += reward * (self.gamma ** step)
                state = next_state
                step += 1
            returns_1.append(return_1)
        
        return_diff = np.array(returns_0) - np.array(returns_1)

        if is_save_metadata:
            data = pd.DataFrame({
                'return_0': returns_0,
                'return_1': returns_1,
                'return_diff': return_diff})
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            data.to_csv(save_path, index=False)

        return returns_0, returns_1, return_diff

    def dist_estimation(
            self,
            n_samples: int = 20,
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> tuple[float, float]:
        
        t_statistics = []
        for _ in range(n_samples):
            returns_0, returns_1, return_diff = self.sample(test_map, test_prob,)
            t = self.stat_const('paired_t', returns_0, returns_1)
            t_statistics.append(t)
        print('data collection finished!')
        # sns.histplot(t_statistics, kde=False, stat='density', alpha=0.5, edgecolor='black', linewidth=0.5,)
        # plt.show()

        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(t_statistics))
        # kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(t_statistics) - 1, max(t_statistics) + 1, 1000)
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, is_plot=True)

        return reject_region

    def run_test(
            self,
            reject_region: tuple[float, float],
            test_map: int = 0, 
            test_prob: float = 0.9,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        if is_map_test is None:
            raise ValueError('Please specify the test type!')
        returns_0, returns_1, return_diff = self.sample(test_map=test_map, test_prob=test_prob, 
                                is_save_metadata=True,)
        t = self.stat_const('paired_t', returns_0, returns_1)
        if t < reject_region[0] or t > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False

    def power_analysis(
            self,
            reject_region, 
            test_map: int = 0, 
            test_prob: float = 0.9, 
        ) -> float:
        
        results = []
        for _ in range(100):
            results.append(self.run_test(reject_region=reject_region, test_map=test_map, 
                                         test_prob=test_prob))
        power = sum(results) / 100
        return power


    
if __name__ == "__main__":
    
    is_map_test = False
    n_MC_samples = 1000
    
    prob = np.arange(0.9, 0.1, -0.01)
    map_list = range(7)

    powers = []

    save_dir = 'data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    policy_dir = 'save/model/policy/map_0_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]

    tester = maze_return_tester(is_map_test=is_map_test, policy=policy, n_trials=10, gamma=0.99)
    reject_region = tester.dist_estimation(n_samples=1000, )

    if is_map_test:
        for m in map_list:
            power = tester.power_analysis(reject_region, test_map=m,)
            powers.append(power)
            print(f'map: {m}, power: {power:.3f}')
        data = pd.DataFrame({
        'map_id': map_list, 
        'power': powers,
        })
        save_path = f'{save_dir}/return_map_test_results.csv'
    else:
        for p in prob:
            power = tester.power_analysis(reject_region, test_prob=p,)
            powers.append(power)
            print(f'prob: {p}, power: {power:.3f}')
        data = pd.DataFrame({
        'prob': prob, 
        'power': powers,
        })
        save_path = f'{save_dir}/return_prob_test_results.csv'
    
    data.to_csv(save_path, index=False)