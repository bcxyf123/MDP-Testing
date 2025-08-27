"""
This module contains classes and functions for testing and estimating probabilities in a maze environment.

Classes:
- maze_return_tester: A class for testing return differences between two environments in a maze.
- maze_lrt_tester: A class for performing likelihood ratio tests in a maze environment.

Functions:
- collect_transitions: Collects state-action-next_state transitions from an environment using a given policy.
- action_transform_batch: Transforms a batch of actions using a predefined action map.
- estimate_prob: Estimates the probabilities of different types of state transitions in an environment using a given policy.
- map_test: Performs return difference tests between two environments with different maps in a maze.
- prob_test: Performs likelihood ratio tests between two environments with different probabilities in a maze.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from utils.utils import *
from envs.gym_simplegrid.generator import genEnv
from base import return_tester, lrt_tester
from scipy.stats import gaussian_kde
...
import numpy as np
import pandas as pd
from typing import List, Tuple

from utils.utils import *
from envs.gym_simplegrid.generator import genEnv

from base import t_tester, lrt_tester
from scipy.stats import gaussian_kde

# utility function
def collect_transitions(policy, env, episodes):
    transitions = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _, info = env.step(action)
            transitions.append((state, action, next_state))
            state = next_state
    return transitions

def action_transform_batch(actions):
    action_map = np.array([-8, 8, -1, 1])
    # actions = np.array(actions)
    return action_map[actions]

def estimate_prob(policy, env, num_epsisodes=50):

    transitions = collect_transitions(policy, env, episodes=num_epsisodes)
    states, actions, next_states = zip(*transitions)
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    delta_states = next_states - states
    
    expected_delta_states = action_transform_batch(actions)
    probs = np.zeros(5)

    # Compare expected and actual delta states
    equal_states = expected_delta_states == delta_states
    opposite_states = expected_delta_states == -delta_states
    zero_states = delta_states == 0

    # Update probabilities based on comparisons
    probs[0] += np.sum(equal_states)
    probs[1] += np.sum(opposite_states)
    probs[-1] += np.sum(zero_states)
    
    rest_prob = len(states) - np.sum(probs)
    probs[2] += rest_prob / 2
    probs[3] += rest_prob / 2
    
    probs = probs / len(states)
    # print(sum(probs))

    return probs


# testers
class maze_return_tester(t_tester):
    """
    A class for testing the return difference between two environments in a maze scenario.

    Args:
        policy: The policy used for action selection.
        gamma (float): The discount factor for computing returns. Default is 0.99.

    Attributes:
        policy: The policy used for action selection.
        gamma (float): The discount factor for computing returns.
        default_map (int): The default map ID.
        default_prob (float): The default probability.
    """

    def __init__(self, policy, gamma=0.99):
        super().__init__()
        self.policy = policy
        self.gamma = gamma
        self.default_map = 0
        self.default_prob = 0.9

    def sample(
            self,
            n_trials: int = 10,
            test_map: int = 0,
            test_prob: float = 0.9,
            is_save_metadata: bool = False,
            save_path: str = None,
        ) -> np.ndarray:
        """
        Sample the return difference between two environments.

        Args:
            n_trials (int): The number of trials to run. Default is 10.
            test_map (int): The map ID for the test environment. Default is 0.
            test_prob (float): The probability for the test environment. Default is 0.9.
            is_save_metadata (bool): Whether to save the metadata. Default is False.
            save_path (str): The path to save the metadata. Required if is_save_metadata is True.

        Returns:
            np.ndarray: The array of return differences.

        Raises:
            ValueError: If is_save_metadata is True but save_path is not specified.
        """
        if is_save_metadata and save_path is None:
            raise ValueError('Please specify the save directory!')
    
        test_map = self.default_map
        test_prob = self.default_prob

        default_env = genEnv(map_size=(4,8), map_id=self.default_map, max_prob=self.default_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
        test_env = genEnv(map_size=(4,8), map_id=test_map, max_prob=test_prob, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

        # compute returns 0
        returns_0 = compute_returns(self.policy, default_env, gamma=self.gamma, iterations=n_trials)
        # compute returns 1
        returns_1 = []
        for _ in range(n_trials):
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
        
        returns_diff = np.array(returns_0) - np.array(returns_1)

        if is_save_metadata:
            data = pd.DataFrame({
                'return_0': returns_0,
                'return_1': returns_1,
                'return_diff': returns_diff})
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path, index=False)

        return returns_0, returns_1, returns_diff

    def dist_estimation(
            self,
            n_trials: int = 10,
            n_samples: int = 1000,
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> Tuple[float, float]:
        """
        Estimate the distribution of return differences.

        Args:
            n_trials (int): The number of trials to run. Default is 10.
            n_samples (int): The number of samples to collect. Default is 1000.
            test_map (int): The map ID for the test environment. Default is 0.
            test_prob (float): The probability for the test environment. Default is 0.9.

        Returns:
            Tuple[float, float]: The lower and upper bounds of the rejection region.

        Raises:
            None
        """
        t_statistics = []
        for _ in range(n_samples):
            return_diff = self.sample(n_trials=n_trials, test_map=test_map, test_prob=test_prob,)
            t = self.stat_const(return_diff)
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

    def run_single_test(
            self,
            test_type: str,
            reject_region: Tuple[float, float],
            n_trials: int = 10,
            test_map: int = 0, 
            test_prob: float = 0.9,
            is_save: bool = False,
            save_dir: str = None,
        ) -> bool:
        """
        Run a single test and check if the return difference falls within the rejection region.

        Args:
            reject_region (Tuple[float, float]): The rejection region.
            n_trials (int): The number of trials to run. Default is 10.
            test_map (int): The map ID for the test environment. Default is 0.
            test_prob (float): The probability for the test environment. Default is 0.9.
            is_save (bool): Whether to save the metadata. Default is False.
            save_dir (str): The directory to save the metadata. Required if is_save is True.

        Returns:
            bool: True if the return difference falls within the rejection region, False otherwise.

        Raises:
            ValueError: If reject_region is not specified.
            ValueError: If is_save is True but save_dir is not specified.
        """
        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        if is_save and save_dir is None:
            raise ValueError('Please specify the save directory!')
        
        data_0, data_1, _ = self.sample(n_trials=n_trials, test_map=test_map, test_prob=test_prob, 
                                is_save_metadata=is_save, save_dir=save_dir)
        t = self.stat_const(test_type=test_type, data_0=data_0, data_1=data_1)
        if t < reject_region[0] or t > reject_region[1]:
            return True
        else:
            return False

    def power_analysis(
            self,
            test_type: str,
            reject_region: Tuple[float, float], 
            test_map: int = 0, 
            test_prob: float = 0.9, 
        ) -> float:
        """
        Perform power analysis to estimate the statistical power.

        Args:
            reject_region (Tuple[float, float]): The rejection region.
            test_map (int): The map ID for the test environment. Default is 0.
            test_prob (float): The probability for the test environment. Default is 0.9.

        Returns:
            float: The estimated statistical power.

        Raises:
            None
        """
        results = []
        for _ in range(100):
            results.append(self.run_single_test(test_type=test_type, reject_region=reject_region, test_map=test_map, 
                                         test_prob=test_prob))
        power = sum(results) / 100
        return power


def map_test(map_id=0, n_eps=10):
    
    policy_dir = 'save/model/policy/map_0_policy_2000.csv'
    default_model_dir = 'save/model/forward_model/forward_model_default.pth'
    test_model_dir = f'save/model/forward_model/forward_model_{map_id}.pth'

    policy = read_csv(policy_dir)[:, -1]
    default_model = TransitionModel(obs_dim=1, action_dim=1)
    default_model.load_state_dict(torch.load(default_model_dir))
    test_model = TransitionModel(obs_dim=1, action_dim=1)
    test_model.load_state_dict(torch.load(test_model_dir))
    
    env = genEnv(map_size=(4,8), map_id=map_id, max_prob=0.9, start_loc=(3,0), goal_loc=(0,7), deterministic=False)

    trajectories = collect_trajectories(env, policy, episodes=n_eps)
    model0 = default_model
    model1 = test_model

    likelihood0_list = []
    likelihood1_list = []
    ratios = []

    for trj in trajectories:
        likelihood0, likelihood1, ratio = likelihoodRatio(trj, model0, model1)
        likelihood0_list.append(likelihood0)
        likelihood1_list.append(likelihood1)
        ratios.append(ratio)
    
    # power = chi_test(ratios, df=2)

    # print(f'likelihood0: {np.mean(likelihood0_list):.3f} +- {np.std(likelihood0_list):.3f}')
    # print(f'likelihood1: {np.mean(likelihood1_list):.3f} +- {np.std(likelihood1_list):.3f}')
    # print(f'ratio: {np.mean(ratios):.3f} +- {np.std(ratios):.3f}')
    # print(f'power: {power:.2f}')

    return ratios


def prob_test(test_p, n_eps=10, es_size=10):
    default_p = 0.9
     # load trained policy
    policy_path = f'save/model/policy/map_0_policy_2000.csv'
    policy = read_csv(policy_path)[:, -1]
    # default_env
    env_0 = genEnv(map_size=(4,8), map_id=0, max_prob=default_p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    # estimate probability
    probs_0 = estimate_prob(policy, env_0, num_epsisodes=es_size)
    # test env
    env_1 = genEnv(map_size=(4,8), map_id=0, max_prob=test_p, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    # estimate probability
    probs_1 = estimate_prob(policy, env_1, num_epsisodes=es_size)
    # print(probs_1)
    trajs = collect_trajectories(env_1, policy, n_eps)
    # ratios = abs(np.array(lrt(trajs, probs_0, probs_1)))
    ratio_list = lrt(trajs, probs_0, probs_1)
    # power = chi_test(ratios, df=5)
    # powers.append(power)
    # plt.hist(ratios, bins='auto', alpha=0.7, color='blue', edgecolor='black', density=True, label='likelihood ratio')
    # plt.show()
    # print()
    # plt.plot(p_list, powers)
    # plt.show()
    return ratio_list


class maze_lrt_tester(lrt_tester):
    def __init__(self, is_map_test, policy, n_trials=10, gamma=0.99):
        super().__init__()
        self.policy = policy
        self.gamma = gamma
        self.default_map = 0
        self.default_prob = 0.9

    def sample(
            self,
            test_map: int = 0,
            test_prob: float = 0.9,
            is_save_metadata: bool = False,
            save_path: str = None,
        ) -> np.ndarray:
        if self.is_map_test:
            ratio_list = map_test(map_id=test_map, n_eps=self.n_trials)
        else:
            ratio_list = prob_test(test_p=test_prob, n_eps=self.n_trials, es_size=50)

        return np.array([ratio_list])
    
    def dist_estimation(
            self, 
            n_samples: int, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> Tuple[float, float]:

        chi_stats = []
        for _ in range(n_samples):
            if self.is_map_test:
                ratio_arr = self.sample(test_map=test_map,)
            else:
                ratio_arr = self.sample(test_prob=test_prob,)
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
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, is_plot=True)

        return reject_region
    
    def run_test(
            self, 
            reject_region: float, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> bool:

        if reject_region is None:
            raise ValueError('Please specify the rejection region!')
        if self.is_map_test:
            ratio_arr = self.sample(test_map=test_map, is_save_metadata=True,)
        else:
            ratio_arr = self.sample(test_prob=test_prob, is_save_metadata=True,)
        chi_stat = np.mean(ratio_arr)
        if chi_stat < reject_region[0] or chi_stat > reject_region[1]:
            # print(f't: {t:.4f}, reject')
            return True
        else:
            # print(f't: {t:.4f}, accept')
            return False
    
    def power_analysis(
            self, 
            reject_region: float, 
            test_map: int = 0,
            test_prob: float = 0.9,
        ) -> float:
        
        powers = []
        for _ in range(100):
            if self.is_map_test:
                powers.append(self.run_test(reject_region, test_map=test_map,))
            else:
                powers.append(self.run_test(reject_region, test_prob=test_prob,))
        power = sum(powers) / 100

        return power
