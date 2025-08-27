from typing import List, Tuple

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind

import abc


class tester(abc.ABC):
    def __init__(self, ):
        """
        Base class for testers.
        """
        pass

    @abc.abstractmethod
    def stat_const(
            self,
            data: list,
        ) -> np.ndarray:
        """
        Compute the statistics of the default policy.

        Parameters:
        data (list): The input data.

        Returns:
        stats (np.ndarray): The statistics of the default policy.
        """
        raise NotImplementedError("stat_const method is not implemented!")
    
    def sample(
            self,
            n_trials: int,
            is_save_metadata: bool = False,
        ) -> np.ndarray:
        """
        Sample the statistics of the default policy.

        Parameters:
        is_save_metadata (bool, optional): Whether to save metadata. Defaults to False.

        Returns:
        stats (np.ndarray): The statistics of the default policy.
        """
        raise NotImplementedError("sample method is not implemented!")

    def kde_estimation(
            self,
            data: np.ndarray
        ) -> gaussian_kde:
        """
        Perform kernel density estimation on the given data and plot the results.

        Parameters:
        data (np.ndarray): The input data for kernel density estimation.

        Returns: 
        kde (gaussian_kde): The kernel density estimate.
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
        ) -> Tuple[float, float]:
        """
        Calculates the rejection region for a given kernel density estimate (KDE) and x-grid.

        Parameters:
            kde (gaussian_kde): The kernel density estimate.
            x_grid (np.ndarray): The x-grid values.
            alpha (float, optional): The significance level. Defaults to 0.05.
            is_plot (bool, optional): Whether to plot the KDE with the rejection region. Defaults to False.

        Returns:
            rejection_region (Tuple[float, float]): The lower and upper bounds of the rejection region.
        """
        # density estimate
        kde_values = kde.evaluate(x_grid)

        # cdf
        cdf_values = np.cumsum(kde_values) * (x_grid[1] - x_grid[0])

        # quantiles
        lower_alpha = alpha / 2
        upper_alpha = 1 - alpha / 2

        # lower and upper indices
        lower_index = np.where(cdf_values > lower_alpha)[0][0]
        upper_index = np.where(cdf_values > upper_alpha)[0][0]

        # lower and upper bounds
        lower_bound = x_grid[lower_index]
        upper_bound = x_grid[upper_index]

        print(f'rejection region: {lower_bound:.4f}, {upper_bound:.4f}')

        if is_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(x_grid, kde_values, label='KDE', color='blue')
            plt.fill_between(x_grid, kde_values, where=(x_grid < lower_bound) | (x_grid > upper_bound), color='red', alpha=0.5, label='Rejection Region')
            plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Critical Value')
            plt.axvline(upper_bound, color='red', linestyle='--', label='Upper Critical Value')
            plt.legend()
            plt.xlabel('Data values')
            plt.ylabel('Density')
            plt.savefig('kde.pdf', dpi=300)

        return lower_bound, upper_bound
    
    def dist_estimation(
            self,
            n_samples: int,
        ) -> Tuple[float, float]:
        """
        Estimate the distribution of the difference in returns between the default and the test.

        Parameters:
        n_samples (int): The number of samples.

        Returns:
        rejection_region (Tuple[float, float]): The rejection region of the estimated distribution.
        """
        raise NotImplementedError("dist_estimation method is not implemented!")

    def run_single_test(
            self,
            n_trials,
            reject_region, 
        ) -> bool:
        """
        Run a hypothesis test to compare the returns of the default and test policies.

        Parameters:
        reject_region (Tuple[float, float]): The rejection region of the estimated distribution.

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
        reject_region (Tuple[float, float]): The rejection region of the estimated distribution.

        Returns:
        float: The statistical power of the hypothesis test.
        """
        raise NotImplementedError("compute_power method is not implemented!")


class t_tester(tester):
    def __init__(self, n_trials=10):
        """
        Tester class for return testing.

        Parameters:
        n_trials (int, optional): The number of trials. Defaults to 10.
        save_dir (str, optional): The directory to save data. Defaults to 'data'.
        """
        super(t_tester, self).__init__()
        self.n_trials = n_trials

    def paired_t(
            self,
            data_1: list,
            data_2: list,
        ) -> float:

        if len(data_1) != len(data_2):
            raise ValueError('two datasets must match in paired-t test!')
        
        diff = np.array(data_1) - np.array(data_2)
        diff_mean = np.mean(diff)
        diff_std = np.std(diff, ddof=1)
        t_stat = diff_mean / (diff_std / np.sqrt(len(diff)))
    
        return t_stat

    def independent_t(
            self,
            data_1: list,
            data_2: list,
        ) -> float:
        t_stat, _ = ttest_ind(data_1, data_2)

        return t_stat

    def choose_test_type(
            self,
            test_type: str,
        ) -> callable:
        if test_type == 'paired_t':
            return lambda data_1, data_2: self.paired_t(data_1, data_2)
        elif test_type == 'independent_t':
            return lambda data_1, data_2: self.independent_t(data_1, data_2)
        else:
            raise ValueError('Unknown test type! Select from "paired_t" or "independent_t".')

    def stat_const(
            self,
            test_type: str,
            data_1: list,
            data_2: list,
        ) -> float:
        """
        Compute the statistics of the default policy for return testing.

        Parameters:
        data (list): The input data.

        Returns:
        t (float): The statistics of the default policy.
        """
        # data_mean = np.mean(data)
        # data_var = np.sum((data - data_mean)**2) / (len(data)-1)
        # t = data_mean / np.sqrt(data_var/len(data))

        test = self.choose_test_type(test_type)
        t_stat = test(data_1, data_2)

        return t_stat


class lrt_tester(tester):
    def __init__(self, ):
        """
        Tester class for likelihood ratio testing.

        Parameters:
        n_trials (int, optional): The number of trials. Defaults to 1.
        save_dir (str, optional): The directory to save data. Defaults to 'data'.
        """
        super(lrt_tester, self).__init__()

    def lrt(
            self,
            data_1: list,
            data_2: list,
        ) -> float:
        
        return np.mean(data_2/data_1)

    def choose_test_type(
            self,
            test_type: str,
        ) -> callable:
        if test_type == 'lrt':
            return lambda data_1, data_2: self.lrt(data_1, data_2)
        else:
            raise ValueError('Unknown test type! Select from "lrt".')

    def stat_const(
            self,
            test_type: str,
            data_1: list,
            data_2: list,
        ) -> float:
        """
        Compute the statistics of the default policy for likelihood ratio testing.

        Parameters:
        data (list): The input data.

        Returns:
        mean (float): The statistics of the default policy.
        """
        test = self.choose_test_type(test_type)
        lrt_stat = test(data_1, data_2)

        return lrt_stat

