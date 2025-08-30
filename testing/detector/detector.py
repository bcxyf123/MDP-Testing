import tqdm
import random
import types
import numpy as np
# from gymnasium import Env
from scipy.stats import gaussian_kde, norm
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib
from matplotlib import pyplot as plt

from .agent import Agent
from testing.utils import *
from testing.utils.stats_utils import *
from testing.utils.stats_utils import *
from typing import Type, Dict, Tuple, List  


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']


class Base_Detector(object):
    def __init__(self, save_dir='data'):
        self.save_dir = save_dir

    def sample(
            self,
            is_save_metadata: bool = False,
        ) -> np.ndarray:
        """
        Sample the statistics of the default policy.

        Parameters:
        self-defined based on the specific environment

        Returns:
        stats(np.ndarray): The statistics of the default policy.
        """
        raise NotImplementedError("sample_stats method is not implemented!")

    def stat_const(
            self,
            data: list,
        ) -> np.ndarray:
        """
        Compute the statistics of the default policy.

        Parameters:
        self-defined based on the specific environment

        Returns:
        stats(np.ndarray): The statistics of the default policy.
        """
        raise NotImplementedError("compute_stats method is not implemented!")

    def kde_estimation(
            self,
            data: np.ndarray
        ) -> gaussian_kde:
        """
        Perform kernel density estimation on the given data and plot the results.

        Parameters:
        data (np.ndarray): The input data for kernel density estimation.

        Returns: 
        gaussian_kde
        """

        kde = gaussian_kde(data)
        
        x_grid = np.linspace(min(data) - 1, max(data) + 1, 1000)
        kde_values = kde.evaluate(x_grid)

        # plt
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=30, density=True, label='Sample Data Histogram', alpha=0.5)
        plt.plot(x_grid, kde_values, label='KDE Estimate', color='red')
        plt.title('Kernel Density Estimation')
        plt.legend()
        plt.show()

        return kde

    def rejection_region(
            self,
            kde: gaussian_kde,
            x_grid: np.ndarray,
            alpha: float = 0.05,
            is_plot: bool = False,
            test_type: str = 'two-sided',
        ) -> Tuple[float, float]:
        """
        Calculates the rejection region for a given kernel density estimate (KDE) and x-grid, supporting both two-sided and one-sided tests.

        Parameters:
            kde (gaussian_kde): The kernel density estimate.
            x_grid (np.ndarray): The x-grid values.
            alpha (float, optional): The significance level. Defaults to 0.05.
            is_plot (bool, optional): Whether to plot the KDE with the rejection region. Defaults to False.
            test_type (str, optional): The type of test ('two-sided', 'left-tailed', 'right-tailed'). Defaults to 'two-sided'.

        Returns:
            Tuple[float, float]: The lower and upper bounds of the rejection region. For one-sided tests, one of the bounds will be None.

        """
        
        # density estimate
        kde_values = kde.evaluate(x_grid)

        # cdf
        cdf_values = np.cumsum(kde_values) * (x_grid[1] - x_grid[0])

        if test_type == 'two-sided':
            # quantiles for two-sided
            lower_alpha = alpha / 2
            upper_alpha = 1 - alpha / 2

            # lower and upper indices
            lower_index = np.where(cdf_values > lower_alpha)[0][0]
            upper_index = np.where(cdf_values > upper_alpha)[0][0]

            # lower and upper bounds
            lower_bound = x_grid[lower_index]
            upper_bound = x_grid[upper_index]

            print(f'Two-sided rejection region: {lower_bound:.4f}, {upper_bound:.4f}')

        elif test_type == 'left-tailed':
            # quantile for left-tailed
            lower_alpha = alpha

            # lower index
            lower_index = np.where(cdf_values > lower_alpha)[0][0]

            # lower bound
            lower_bound = x_grid[lower_index]
            upper_bound = float('inf')

            print(f'Left-tailed rejection region: {lower_bound:.4f}')

        elif test_type == 'right-tailed':
            # quantile for right-tailed
            upper_alpha = 1 - alpha

            # upper index
            upper_index = np.where(cdf_values > upper_alpha)[0][0]

            # upper bound
            upper_bound = x_grid[upper_index]
            lower_bound = float('-inf')

            print(f'Right-tailed rejection region: {upper_bound:.4f}')


        if is_plot:
            # plt.figure(figsize=(8, 6), dpi=300)
            plt.figure()
            plt.plot(x_grid, kde_values, label='KDE', color='blue')
            if test_type == 'two-sided':
                plt.fill_between(x_grid, kde_values, where=(x_grid < lower_bound) | (x_grid > upper_bound), color='red', alpha=0.5, label='Rejection Region')
                plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Critical Value')
                plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Critical Value')
            elif test_type == 'left-tailed':
                plt.fill_between(x_grid, kde_values, where=(x_grid < lower_bound), color='red', alpha=0.5, label='Rejection Region')
                plt.axvline(lower_bound, color='red', linestyle='--', label='Critical Value')
            elif test_type == 'right-tailed':
                plt.fill_between(x_grid, kde_values, where=(x_grid > upper_bound), color='red', alpha=0.5, label='Rejection Region')
                plt.axvline(upper_bound, color='red', linestyle='--', label='Critical Value')
            plt.legend()
            plt.xlabel('Data values')
            plt.ylabel('Density')
            plt.title('Kernel Density Estimation with Rejection Region')
            # plt.savefig(f'{self.save_dir}/rejection_region.pdf')
            plt.show()

        return lower_bound, upper_bound
    
    def dist_estimation(
            self,
            n_samples: int,
        ) -> Tuple[float, float]:
        """
        estimate the distribution of the difference in returns between the default and the test

        Parameters:
        self-defined based on the specific environment

        Returns:
        rejection_region(Tuple): The rejection region of the estimated distribution.
        """
        raise NotImplementedError("dist_estimation method is not implemented!")

    def run_test(
            self,
            reject_region, 
        ) -> bool:
        """
        Run a hypothesis test to compare the returns of the default and test policies.

        Parameters:
        reject_region(Tuple): The rejection region of the estimated distribution.
        self-defined based on the specific environment

        Returns:
        bool: Whether to reject the null hypothesis.
        """
        raise NotImplementedError("run_return_test method is not implemented!")    

    def power_analysis(
            self,
            reject_region, 
        ) -> float:
        """
        Compute the statistical power of the hypothesis test.

        Parameters:
        reject_region(Tuple): The rejection region of the estimated distribution.
        self-defined based on the specific environment

        Returns:
        float: The statistical power of the hypothesis test.
        """
        raise NotImplementedError("compute_power method is not implemented!")



class MDPDetector(Base_Detector):
    def __init__(self, detector_type, data_type, base_agent, gamma=0.99, kernel='rbf', save_dir=None):
        super().__init__()
        
        optional_detectors = ['t', 'Welchs-t', 'mann-whitney-u', 'rank-t', 'lrt', 'mmd']
        optional_data_type = ['return', 'likelihood', 'likelihood-ratio']
        
        assert detector_type in optional_detectors, f"detector_type must be in {optional_detectors}"
        assert data_type in optional_data_type, f"data_type must be in {optional_data_type}"
        assert (detector_type == 'lrt') == (data_type == 'likelihood-ratio'), \
                f"Invalid combination: detector_type is '{detector_type}' and data_type is '{data_type}'"

        
        if detector_type == 'mmd' and kernel is None:
            raise ValueError('Please specify the kernel function for MMD detection!')
        
        self.detector_type = detector_type
        self.data_type = data_type
        
        self.base_agent = base_agent
        self.gamma = gamma
        self.detector = detector_type
        self.data_type = data_type
        self.kernel = kernel
        self.save_dir = save_dir

    
    
    def compute_log_likelihood(
        self,
        env,
        agent: Agent,
        traj: Dict,     # (s, a, r, s')
    ) -> float:
        
        ll = 0.0        # initial value 0 as it is *log* likelihood
        for trans in traj:
            pi_ll = agent.compute_log_likelihood(trans)
            env_ll = env.unwrapped.compute_log_likelihood(trans)
            # print("pi_ll:", pi_ll, "env_ll:", env_ll)
            ll += (pi_ll+env_ll)
        
        return ll


    def sample_const(
            self,
            default_env,
            test_env,
            sample_func: callable,
            num_episodes_default: int,
            num_episodes_test: int,
            estimate_test_env = None,
        ) -> Dict:
        
        if num_episodes_test is None:
            num_episodes_test = num_episodes_default
            raise Warning('The number of test sampels is not specified. Using the same number of default samples.')
        
        if self.data_type in ['return', 'likelihood']:
            raw_samples_0 = sample_func(env=default_env, agent=self.base_agent, gamma=self.gamma, num_episodes=num_episodes_default)        
        raw_samples_1 = sample_func(env=test_env, agent=self.base_agent, gamma=self.gamma, num_episodes=num_episodes_test)        

        if self.data_type == 'return':
            x = raw_samples_0['returns']
            y = raw_samples_1['returns']
        elif self.data_type == 'likelihood':
            x = []
            y = []
            for traj0, traj1 in zip(raw_samples_0["trajs"], raw_samples_1["trajs"]):
                ll_0 = self.compute_log_likelihood(default_env, self.base_agent, traj0)
                ll_1 = self.compute_log_likelihood(default_env, self.base_agent, traj1)
                x.append(ll_0)
                y.append(ll_1)
        elif self.data_type == 'likelihood-ratio':
            x = []
            y = []
            for traj1 in raw_samples_1["trajs"]:
                ll_0 = self.compute_log_likelihood(default_env, self.base_agent, traj1)
                ll_1 = self.compute_log_likelihood(test_env, self.base_agent, traj1)
                x.append(ll_0 - ll_1)
                diff = ll_0 - ll_1
                # print(f"ll_0: {ll_0:.4f}, ll_1: {ll_1:.4f}, diff: {diff:.4f}")
                y.append(0.0)  # dummy y (not used)
        else:
            raise ValueError('Unknown data type!')
                
        return x, y
        

    def stat_const(
            self,
            x: List,
            y: List,
        ) -> float:
        
        ## paired-t
        # if self.detector == 't':
        #     data = np.array(x) - np.array(y)
        #     data_mean = np.mean(data)
        #     data_var = np.sum((data - data_mean)**2) / (len(data)-1)
        #     t = data_mean / np.sqrt(data_var/len(data))
        #     stat = t
        
        if self.detector == 't':
            stat = compute_t(x, y)
            
        elif self.detector == 'Welchs-t':
            stat = compute_welch_t(x, y)
            
        elif self.detector == 'mann-whitney-u':
            stat = compute_mann_whitney_u(x, y)
            
        elif self.detector == 'rank-t':
            stat = compute_rank_t(x, y)
        
        elif self.detector == 'lrt':
            stat = compute_lr(x)
                    
        elif self.detector == 'mmd':
            stat = compute_mmd(x, y, kernel=self.kernel)
            
        else:
            raise ValueError('Unknown detector type!')

        return stat


    # construct distribution for null hypothesis
    def dist_estimation(
            self,
            default_env,
            sample_func: callable, 
            n_samples: int = 1000,
            num_episodes_default: int = 10,
            num_episodes_test: int = None,
            sample_method: str = 'MC',
        ) -> Tuple[float, float]:

        if num_episodes_test is None:
            num_episodes_test = num_episodes_default
            raise Warning('The number of test sampels is not specified. Using the same number of default samples.')
        
        stats = []
        # MC sampling
        if sample_method == 'MC':
            for _ in tqdm.tqdm(range(n_samples), desc='Running MC sampling for dist estimation under H0'):
                x, y = self.sample_const(default_env=default_env, test_env=default_env, sample_func=sample_func, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test)
                t = self.stat_const(x, y)
                stats.append(t)
        # bootstrap sampling
        elif sample_method == 'bootstrap':
            x, y = self.sample_const(default_env=default_env, test_env=default_env, sample_func=sample_func, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test)
            N_X = len(x)
            N_Y = len(y)
            xy = x + y
            for i in tqdm.tqdm(range(n_samples), desc='Running bootstrap sampling for dist estimation under H0'):
                random.shuffle(xy)  
                t = self.stat_const(xy[:N_X], xy[N_X:])
                stats.append(t)

        print(f'mean: {np.mean(stats):.2f}, std: {np.std(stats):.2f}')
        # print("Sampled LRT stats head:", stats[:10])

        # --------- Fix KDE error handling ----------
        if np.std(stats) < 1e-8:
            print("Warning: All statistics are the same, cannot fit KDE. Using a tiny rejection region around the constant value.")
            constant = stats[0]
            eps = 1e-6
            return (constant - eps, constant + eps)


        # # use kernel density estimation to estimate the distribution
        kde = gaussian_kde(np.array(stats))
        # kde_estimation(np.array(t_statistics))
        # pdf range
        x_grid = np.linspace(min(stats) - 1, max(stats) + 1, 1000)

        if self.detector in ['mmd', 'lrt']:
            test_type = 'right-tailed'
        else:
            test_type = 'two-sided'
            
        reject_region = self.rejection_region(kde, x_grid, alpha=0.05, test_type=test_type, is_plot=False)

        return reject_region


    def run_test(
            self,
            reject_region: Tuple[float, float],
            default_env,
            test_env,
            sample_func: callable,
            num_episodes_default: int=10,
            num_episodes_test: int=None
        ) -> bool:
        
        if num_episodes_test is None:
            num_episodes_test = num_episodes_default
            raise Warning('The number of test sampels is not specified. Using the same number of default samples.')

        x, y = self.sample_const(default_env=default_env, test_env=test_env, sample_func=sample_func, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test)
        t = self.stat_const(x, y)
        if t < reject_region[0] or t > reject_region[1]:
            return True
        else:
            return False


    # run 100 times testing to compute power
    def power_analysis(
            self,
            reject_region: Tuple[float, float],
            default_env,
            test_env,
            sample_func: callable,
            num_episodes_default: int=10,
            num_episodes_test: int=None
        ) -> float:
        
        if num_episodes_test is None:
            num_episodes_test = num_episodes_default
            raise Warning('The number of test sampels is not specified. Using the same number of default samples.')
        
        # reject_region = self.dist_estimation(default_env, sample_func)
        results = []
        for _ in tqdm.tqdm(range(100), desc='Running tests for power analysis'):
            results.append(self.run_test(reject_region=reject_region, default_env=default_env, test_env=test_env, 
                                         sample_func=sample_func, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test))
        power = sum(results) / 100
        return power


