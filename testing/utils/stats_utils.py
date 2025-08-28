import numpy as np
import gymnasium as gym 
import torch
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import ttest_ind, mannwhitneyu, chi2

from typing import List, Tuple


def compute_lr(
    ratio: List,
    ) -> float:
    
    lr = -2 * np.sum(ratio)
    
    return lr


def compute_t(
    x: List,
    y: List,
    ) -> float:
    
    n1, n2 = len(x), len(y)
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled variance for independent t
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    t = (mean1 - mean2) / pooled_std
    
    return t


def compute_welch_t(
    x: List,
    y: List,
    ) -> float:
    
    n1, n2 = len(x), len(y)
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    # Welch's t
    welch_t = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
    
    return welch_t


def compute_mann_whitney_u(
    x: List,
    y: List,
    ) -> float:
    
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1  # 排序后分配秩值
    ranks_x1 = ranks[:n1]
    ranks_x2 = ranks[n1:]
    R1, R2 = np.sum(ranks_x1), np.sum(ranks_x2)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = R2 - n2 * (n2 + 1) / 2
    U = min(U1, U2)
    
    return U

def compute_rank_t(
    x: List,
    y: List,
    ) -> float:
    
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1  # 排序后分配秩值
    ranks_x1 = ranks[:n1]
    ranks_x2 = ranks[n1:]
    t = compute_t(ranks_x1, ranks_x2)
    
    return t

def compute_permutation(
    x: List,
    y: List,
    ) -> float:
    
    pass

def compute_mmd(
    x: List,
    y: List,
    kernel: str = "rbf",
    ) -> float:
    
    x = torch.tensor(x).to(torch.float)
    y = torch.tensor(y).to(torch.float)
    if x.dim() == 1:
        x = x[:, None]
    if y.dim() == 1:
        y = y[:, None]
        
    # sigma for mmd kernel
    def compute_sigma(x, y):
        dists = torch.pdist(torch.cat([x, y], dim=0))
        sigma = dists.median()/2

        return sigma.detach().item()
    
    sigma = compute_sigma(x, y)
    XX = pairwise_kernels(x, x, metric=kernel, gamma=sigma)
    YY = pairwise_kernels(y, y, metric=kernel, gamma=sigma)
    XY = pairwise_kernels(x, y, metric=kernel, gamma=sigma)

    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    mmd = np.maximum(mmd, 0)
    stat = np.sqrt(mmd)

    return stat


# tests
def ind_t_test(
    x: List,
    y: List,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    t_stat, p_value = ttest_ind(x, y, equal_var=True)  # 假设方差相等
    if p_value < alpha:
        return True
    else:
        return False
    
def welch_t_test(
    x: List,
    y: List,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    t_stat, p_value = ttest_ind(x, y, equal_var=False)  # 假设方差不等
    if p_value < alpha:
        return True
    else:
        return False
    
def mann_whitney_u_test(
    x: List,
    y: List,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    u_stat, p_value = mannwhitneyu(x, y, alternative='two-sided')
    if p_value < alpha:
        return True
    else:
        return False
    
def rank_t_test(
    x: List,
    y: List,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1  # 排序后分配秩值
    ranks_x1 = ranks[:n1]
    ranks_x2 = ranks[n1:]
    
    t_stat, p_value = ttest_ind(ranks_x1, ranks_x2)
    
    if p_value < alpha:
        return True
    else:
        return False
    
    
def permutation_test(
    x: List,
    y: List,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    pass

def lrt_test(
    x: List,
    y: List,
    df_diff: int,
    alpha: float = 0.05,
    ) -> Tuple[float, float]:
    
    if len(y) == 0:
        ratio = -2 * x
    else:   
        ratio = -2 * (x-y)
    
    p_value = 1 - chi2.cdf(ratio, df_diff)
    if p_value < alpha:
        return True
    else:
        return False