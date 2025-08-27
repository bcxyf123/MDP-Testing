import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from scipy.stats import ttest_rel
from scipy.stats import beta
from scipy.stats import gaussian_kde

from utils.utils import *
from envs.bandit.bandit import BernoulliBandit
from algo.ThompsonSampling import ThompsonSampling

from base import return_tester, lrt_tester


class bandit_return_tester(return_tester):
    def __init__(self, n_trials=10, default_prob=0.9, n_steps=500,):
        super().__init__(
            n_trials=n_trials,
        )
        self.default_p = default_prob
        self.n_steps = n_steps

    def sample(
            self,
            test_p: float = 0.9, 
            is_save_metadata: bool = False,
            save_dir: str = 'data',
        ) -> np.ndarray:
        
        default_bandit = BernoulliBandit(p=self.default_p)
        default_solver = ThompsonSampling(default_bandit)
        test_bandit = BernoulliBandit(p=test_p)
        test_solver = ThompsonSampling(test_bandit)

        return_0 = []
        return_1 = []

        default_solver.run(self.n_steps)
        test_solver._a = default_solver._a
        test_solver._b = default_solver._b
        default_solver.policy_fixed = True
        test_solver.policy_fixed = True
        default_solver.reset()

        for _ in range(self.n_trials):
            
            test_solver.run(self.n_steps, update=False)
            default_solver.run(self.n_steps, update=False)

            rewards_0 = []
            actions = test_solver.actions

            for a in actions:
                # call default env
                rewards_0.append(default_bandit.step(a))
            return_0.append(np.mean(rewards_0))

            # return(r_1(s_1))
            return_1.append(np.mean(test_solver.rewards))

            default_solver.reset()
            test_solver.reset()

        return_diff = np.array(return_0) - np.array(return_1)

        if is_save_metadata:
            data = pd.DataFrame({
                'return_0': return_0,
                'return_1': return_1,
                'return_diff': return_diff})
            # save_dir = 'data'
            save_path = f'{save_dir}/return_test_metadata_{test_p}.csv'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # automatically create or overwrite
            data.to_csv(save_path, index=False)

        return return_diff

    def dist_estimation(
            self,
            n_samples: int = 1000, 
        ) -> Tuple[float, float]:

        t_stats = []
        for _ in range(n_samples):
            return_diff = self.sample(test_p=self.default_p,)
            t = self.stat_const(return_diff)
            t_stats.append(t)
        print('data collection finished!')
        # sns.histplot(t_statistics, kde=False, stat='density', alpha=0.5, edgecolor='black', linewidth=0.5,)
        # plt.show()

        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(t_stats))
        # kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(t_stats) - 1, max(t_stats) + 1, 1000)
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, is_plot=True)

        return reject_region

    def run_test(
            self,
            reject_region: Tuple[float, float],
            test_p: float = 0.9, 
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        return_diff = self.sample(test_p=test_p, is_save_metadata=True,)
        t = self.stat_const(return_diff)
        if t < reject_region[0] or t > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False

    def power_analysis(
            self,
            reject_region: Tuple[float, float],
            test_p: float = 0.9, 
        ) -> float:
        
        powers = []
        for _ in range(100):
            powers.append(self.run_test(reject_region, test_p, ))
        power = sum(powers) / 100
        return power

class bandit_lrt_tester(lrt_tester):
    def __init__(self, n_trials=10, default_prob=0.9, n_steps=500,):
        super().__init__(n_trials,)
        self.default_p = default_prob
        self.n_steps = n_steps

    def sample(
            self, 
            is_save_metadata: bool = False,
            test_p: float = 0.9,
        ) -> np.ndarray:
        
        default_bandit = BernoulliBandit(p=self.default_p)
        default_solver = ThompsonSampling(default_bandit)
        test_bandit = BernoulliBandit(p=test_p)
        test_solver = ThompsonSampling(test_bandit)

        ratio_list = []
        log_ratio_list = []
        likelihood0_list = []
        likelihood1_list = []

        default_solver.run(self.n_steps)
        test_solver._a = default_solver._a
        test_solver._b = default_solver._b
        default_solver.policy_fixed = True
        test_solver.policy_fixed = True
        default_solver.reset()

        for _ in range(self.n_trials):

            test_solver.run(self.n_steps, update=False)
            # print(test_solver._a, test_solver._b)
            # print(test_solver._a_bar, test_solver._b_bar)
            actions = test_solver.actions
            rewards = test_solver.rewards
            # use default env model
            likelihood0 = (sum(np.log(default_bandit.probs[a] if r == 1 else 1 - default_bandit.probs[a]) 
                            for a, r in zip(actions, rewards) if r in [0, 1]))
            # use frequency estimation
            likelihood1 = (sum(np.log(test_solver.calculate_bernoulli_pdf(a, r)) 
                            for a, r in zip(actions, rewards) if r in [0, 1]))
            likelihood0_list.append(likelihood0)
            likelihood1_list.append(likelihood1)
            ratio = -2*(likelihood0 - likelihood1)
            if ratio == 0:
                ratio += 1e-10
            ratio_list.append(ratio)
            log_ratio_list.append(np.log(ratio))
            test_solver.reset()

        return np.array(ratio_list)
    
    def dist_estimation(
            self, 
            n_samples: int
        ) -> Tuple[float, float]:

        chi_stats = []
        for _ in range(n_samples):
            ratio_arr = self.sample(test_p=self.default_p,)
            chi_stat = np.mean(ratio_arr)
            chi_stats.append(chi_stat)
        print('data collection finished!')
        # sns.histplot(t_statistics, kde=False, stat='density', alpha=0.5, edgecolor='black', linewidth=0.5,)
        # plt.show()

        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(chi_stats))
        # kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(chi_stats) - 1, max(chi_stats) + 1, 1000)
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, )

        return reject_region

    def run_test(
            self, 
            reject_region: Tuple[float, float],
            test_p: float,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        ratio_arr = self.sample(test_p=test_p, is_save_metadata=True,)
        chi_stat = np.mean(ratio_arr)
        if chi_stat < reject_region[0] or chi_stat > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False
    
    def power_analysis(
            self, 
            reject_region: Tuple[float, float],
            test_p: float,
        ) -> float:

        powers = []
        for _ in range(100):
            powers.append(self.run_test(reject_region, test_p, ))
        power = sum(powers) / 100

        return power



