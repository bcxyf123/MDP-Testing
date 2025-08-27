import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gym_detect import Parameter_name_list, genEnv
from train_transition_model import CARL_obs_transform
from utils.utils import *

class gym_return_tester(return_tester):
    def __init__(self, policy, num_trajs=10):
        super().__init__()
        self.policy = policy
        self.num_trajs = num_trajs

    def sample(
            self,
            env_name: str,
            param_name: str,
            test_id: int = 0,
            is_save_metadata: bool = False,
        ) -> np.ndarray:

        env_0, contexts, default_id = genEnv(env_name, param_name)
        env_1, _, _ = genEnv(env_name, param_name)

        env_0.context_id = 0
        env_0.reset()

        env_1.context_id = test_id
        env_1.reset()

        returns_0 = []
        returns_1 = []
        
        for _ in range(self.num_trajs):
            obs_0 = env_0.reset()
            obs_0 = CARL_obs_transform(obs_0)
            done_0 = False
            truncated_0 = False
            return_0 = 0 
            steps_0 = 0
            while not (done_0 or truncated_0):
                steps_0 += 1
                action_0, _ = self.policy.predict(obs_0, deterministic=False)
                new_obs_0, reward_0, done_0, truncated_0, info_0 = env_0.step(action_0)
                new_obs_0 = CARL_obs_transform(new_obs_0)
                obs_0 = new_obs_0
                return_0 += reward_0*(0.99**steps_0)
            returns_0.append(return_0)

            obs_1 = env_1.reset()
            obs_1 = CARL_obs_transform(obs_1)
            done_1 = False
            truncated_1 = False
            return_1 = 0 
            steps_1 = 0
            while not (done_1 or truncated_1):
                steps_1 += 1
                action_1, _ = self.policy.predict(obs_1, deterministic=False)
                new_obs_1, reward_1, done_1, truncated_1, info_1 = env_1.step(action_1)
                new_obs_1 = CARL_obs_transform(new_obs_1)
                obs_1 = new_obs_1
                # r_0(s_1)
                reward = reward_func(env_name, obs_1)
                return_1 += reward*(0.99**steps_1)
            returns_1.append(return_1)

        return_diff = np.array(returns_0) - np.array(returns_1)

        # save_dir = f'tables/{env_name}/{param_name}/value_test.csv'
        # if not os.path.exists(os.path.dirname(save_dir)):
        #     os.makedirs(os.path.dirname(save_dir))

        # data = {
        #     'return_0_means': return_0_mean,
        #     'return_1_means': return_1_mean,
        #     'return_diff_means': return_diff_mean,
        #     'return_0_stds': return_0_std,
        #     'return_1_stds': return_1_std,
        #     'return_diff_stds': return_diff_std,
        #     'powers': powers
        # }
        # df = pd.DataFrame(data)
        # df.to_csv(save_dir, index=False)

        return return_diff
    

    def dist_estimation(
            self,
            n_samples: int,
            env_name: str,
            param_name: str,
            test_id: int = 0,
        ) -> Tuple[float, float]:

        t_statistics = []
        for _ in range(n_samples):
            return_diff = self.sample(env_name, param_name, test_id)
            t = self.stat_const(return_diff)
            t_statistics.append(t)
        print('data collection finished!')
        sns.histplot(t_statistics, kde=False, stat='frequency', alpha=0.5, edgecolor='black', linewidth=0.5, label='$\hat{V}_R^{(1)}-\hat{V}_R^{(0)}$')
        plt.show()

        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(t_statistics))
        # self.kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(t_statistics) - 1, max(t_statistics) + 1, 1000)
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, )
        
        return reject_region
    

    def run_test(
            self,
            reject_region: float,
            env_name: str,
            param_name: str,
            test_id: int = 0,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        return_diff = self.sample(env_name, param_name, test_id)
        t = self.stat_const(return_diff)
        if t < reject_region[0] or t > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False
        
    def power_analysis(
            self,
            reject_region: float, 
            env_name: str,
            param_name: str,
            test_id: int = 0,
        ) -> float:

        test_results = []
        for _ in range(100):
            test_results.append(self.run_test(reject_region, env_name, param_name, test_id))
        power = sum(test_results) / 100

        return power

