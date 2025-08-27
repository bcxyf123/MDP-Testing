import os
import sys
import math
sys.path.append('..')
import pandas as pd
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

import numpy as np
from copy import deepcopy
import scipy.stats as stats
from scipy.stats import chi2
from statsmodels.stats.power import tt_solve_power, TTestPower, TTestIndPower, GofChisquarePower
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import abc

from envs.gym_simplegrid.generator import MAP_LIB

# 设置字体为SimHei显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def save_csv(data, filepath):
    '''
    Save data to a CSV file.

    Args:
        data (list or array-like): The data to be saved.
        filepath (str): The path to the CSV file.

    Returns:
        None
    '''
    df = pd.DataFrame(data)
    df.to_csv(filepath)


def read_csv(filepath):
    '''
    Read data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        numpy.ndarray: The data read from the CSV file.
    '''
    df = pd.read_csv(filepath)
    return df.to_numpy()

def write_csv(data, filepath):
    '''
    Write data to a CSV file.

    Args:
    data (dict): A dictionary containing data to be written to the CSV file.
    filepath (str): The path of the CSV file.

    Returns:
    None
    '''
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        df = pd.DataFrame()

    new_data = pd.DataFrame([data])
    df = pd.concat([df, new_data], ignore_index=True)

    df.to_csv(filepath, index=False)


def smooth(data, sm=10):
    d = data
    z = np.ones(len(d))
    y = np.ones(sm)*1.0
    d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
    
    return d



def plot(data, save_path=None, xlabel=None, ylabel=None, title=None):
    '''
    Plot data and save the plot to a file.

    Args:
        data (list or array-like): The data to be plotted.
        save_path (str, optional): The path to save the plot. If not provided, the plot will not be saved.

    Returns:
        None
    '''
    plt.clf()
    plt.plot(data, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def draw_maze(maze, values=None, start=(3, 0), end=(0, 7), save_path=None):
    """
    Draw a maze with optional values in each cell.

    Parameters:
    - maze: 2D numpy array representing the maze.
    - values: Optional 2D numpy array representing values to be displayed in each cell.
    """
    cmap = plt.cm.binary
    norm = plt.Normalize(0, 1)

    fig, ax = plt.subplots()
    plt.imshow(maze, cmap=cmap, norm=norm, interpolation='none', alpha=0.5,)

    top = 'bottom'
    bottom = 'top'
    left = 'right'
    right = 'left'
    center = 'center'

    if values is not None:
        # value_left, value_right, value_up, value_down = values_list
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if values[i, j] is not None and '-' not in values[i, j]:
                    if isinstance(values[i, j], str):
                        values[i, j] = int(values[i, j])
                    value = ['↑', '↓', '←', '→'][values[i, j]]
                    # value = ['上', '下', '左', '右'][values[i, j]]
                else:
                    value = ''
                plt.text(j, i, value, ha=center, va=center, fontsize=30, color='blue')
                # # left
                # ax.text(j, i, str(value_left[i, j]), ha=left, va=center, fontsize=5, color='black')
                # # right
                # ax.text(j, i, str(value_right[i, j]), ha=right, va=center, fontsize=5, color='black')
                # # up
                # ax.text(j, i, str(value_up[i, j]), ha=center, va=top, fontsize=5, color='black')
                # # down
                # ax.text(j, i, str(value_down[i, j]), ha=center, va=bottom, fontsize=5, color='black')

    xticks = [x-0.5 for x in range(maze.shape[1])]
    yticks = [y-0.5 for y in range(maze.shape[0])]
    ax.set_xticks(xticks, minor=False)
    ax.set_yticks(yticks, minor=False)
    plt.grid(which="major", color="black", linestyle='-', linewidth=2)

    # start and goal points
    if start is not None:
        rect_start = Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, linewidth=2, facecolor='red', alpha=0.5)
        ax.add_patch(rect_start)
        # robot_img = mpimg.imread('robot.png')  # Read the robot image
        # plt.imshow(robot_img, extent=(start[0]-0.5, start[0]+0.5, start[1]-0.5, start[1]+0.5),)
    if end is not None:
        rect_end = Rectangle((end[1]-0.5, end[0]-0.5), 1, 1, linewidth=2, facecolor='green', alpha=0.5)
        ax.add_patch(rect_end)
        # trophy_img = mpimg.imread('trophy.png')  # Read the robot image
        # plt.imshow(trophy_img, extent=(end[0]-0.5, end[0]+0.5, end[1]-0.5, end[1]+0.5),)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)

def likelihoodRatio(traj, model0, model1, c=0.5, alpha=0.05):
    
    ratio = 0
    log_likelihood_0 = 0
    log_likelihood_1 = 0

    log_likelihood_0 = compute_trajectory_log_likelihood(traj, model0)
    log_likelihood_1 = compute_trajectory_log_likelihood(traj, model1)

    ratio = -2 * (log_likelihood_0 - log_likelihood_1)
        # if math.isinf(ratio):
        #     print("likelihood0: ", likelihood0)
        #     print("likelihood1: ", likelihood1)
        #     print("ratio: ", ratio)

    return log_likelihood_0, log_likelihood_1, ratio
    # if isinstance(model, nn.Module):
    #     for transition in data:
    #         s, a, r, s_ = transition
    #         next_state_list.append(s_)
    #         with torch.no_grad():
    #             x = torch.tensor(np.array([s, a]), dtype=torch.float32).unsqueeze(0)
    #             output = model(x)
    #         output = output.detach().numpy()[0]
    #         map_output_indices = map_output(s)
    #         if s_ in map_output_indices:
    #             indices = [map_output_indices.index(x) for x in map_output_indices if x == s_]
    #             prob = sum([output[idx] for idx in indices])
    #         else:
    #             prob = 0
    #         prob_list.append(prob)
    #         likelihood += np.log(prob)
    # if isinstance(model, nn.Module):
    #     with torch.no_grad():
    #         states, actions, _, next_states = zip(*data)
    #         states = torch.tensor(states, dtype=torch.float32)
    #         actions = torch.tensor(actions, dtype=torch.float32)
    #         next_states = torch.tensor(next_states, dtype=torch.float32)
    #         # x = torch.cat([states, actions], dim=-1)
    #         mean, std = model(states, actions)
    #         dist = torch.distributions.Normal(mean, std)
    #         log_probs = dist.log_prob(next_states)
    #         log_likelihood = log_probs.sum().item()
    # else:
    #     for transition in data:
    #         s, a, r, s_ = transition
    #         next_state_list.append(s_)
    #         prob = model.get_pfunc(s, a, s_)
    #         prob_list.append(prob)
    #         log_likelihood += np.log(prob)
    # # print(next_state_list)
    # return likelihood


def compute_trajectory_log_likelihood(trajectory, model):
    total_log_prob = 0
    if isinstance(model, nn.Module):
        with torch.no_grad():
            states, actions, _, delta_states = zip(*trajectory)
            states = torch.tensor(np.array(states), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            delta_states = torch.tensor(np.array(delta_states), dtype=torch.float32)

            if model.output_type == 'gaussian':
                means, stds = model(states, actions)
                covs = torch.diag_embed(stds**2)  # Assume independence between dimensions
                dists = torch.distributions.MultivariateNormal(means, covs)
                log_probs = torch.clamp(dists.log_prob(delta_states), min=-10, max=10)
            elif model.output_type == 'categorical':
                logits = model(states, actions)
                dists = torch.distributions.Categorical(logits=logits)
                log_probs = dists.log_prob(delta_states)
            else:
                raise ValueError(f'Unknown output type: {model.output_type}')
            total_log_prob = torch.sum(log_probs).detach().cpu().numpy()
            # print('total_log_prob: ', total_log_prob)
    else:
        # total_log_prob = np.sum(np.log(model.get_pfunc(s, a, s_) for s, a, _, s_ in zip(*trajectory)))
        for transition in trajectory:
            s, a, r, s_ = transition
            if a == -1:
                continue
            prob = model.get_pfunc(s, a, s_)
            total_log_prob += np.log(prob)

    return total_log_prob

def compute_rews(data, model):
    '''
    Compute the reward for a given set of transitions using a model's reward function.

    Args:
    - data (list): A list of transitions, where each transition is a Tuple of (state, action, reward, next_state).
    - model (object): An object with a get_rfunc() method that takes in a state and action and returns the expected reward.

    Returns:
    - rews (float): The sum of the absolute difference between the actual reward and the expected reward for each transition in the data.
    '''
    rews = 0
    for transition in data:
        s, a, r, s_ = transition
        rews += abs(r - model.get_rfunc(s, a))
    return rews/len(data)

def label_states(delta, ncols=8):
    # up
    if delta == -ncols:
        return 0
    # down
    if delta == ncols:
        return 1
    # left
    if delta == -1:
        return 2
    # right
    if delta == 1:
        return 3
    # stay
    if delta == 0:
        return 4

def map_output(state, nrows=4, ncols=8):

    def unlabel_states(y, state, nrows=nrows, ncols=ncols):
        # up
        if y == 0:
            delta = -ncols
            if state < ncols:
                delta = 0
        # down
        if y == 1:
            delta = ncols
            if state >= (nrows-1)*ncols:
                delta = 0
        # left
        if y == 2:
            delta = -1
            if state % ncols == 0:
                delta = 0
        # right
        if y == 3:
            delta = 1
            if state % ncols == ncols-1:
                delta = 0
        # stay
        if y == 4:
            delta = 0
        
        return state + delta
    
    output_indices = [unlabel_states(y, state, ncols) for y in range(5)]
    
    return output_indices

def collect_transitions(policy, env, episodes):
    '''
    Collects data from the environment using the given policy for the specified number of episodes.

    Args:
    policy (dict): A dictionary containing the policy to be used for selecting actions.
    env (gym.Env): The environment to collect data from.
    episodes (int): The number of episodes to collect data for.

    Returns:
    data (list): A list of lists containing the collected data. Each sublist contains the following elements:
                 - obs: The observation at the current time step.
                 - action: The action taken at the current time step.
                 - reward: The reward received at the current time step.
                 - obs_: The observation at the next time step.
    '''
    transitions = []
    for _ in range(episodes):
        obs, _  =  env.reset()
        done = False
        while not done:
            action = policy[obs]
            obs_, reward, done, _, info = env.step(action)
            y = label_states(obs_ - obs)
            transitions.append([obs, action, reward, obs_, y])
            obs = obs_
    return transitions


def collect_trajectory(policy, env, episodes):
    trajectory_list = []
    for _ in range(episodes):
        obs, _  =  env.reset()
        done = False
        trajectory = []
        while not done:
            action = policy[obs]
            obs_, reward, done, _, info = env.step(action)
            trajectory.append([obs, action, reward, obs_])
            obs = obs_
        trajectory_list.append(trajectory)

    return trajectory_list

def t_ind_test(data1, data2):
    t_stat, p_value = stats.ttest_ind(data1, data2)
    # print(f"t-statistic: {t_stat}, p-value: {p_value}")
    
    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    
    effect_size = cohen_d(data1, data2)
    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size=effect_size, nobs1=len(data1), alpha=0.05, ratio=len(data2)/len(data1))
    # print(f'power: {power:.3f}')
    return t_stat, p_value, power


def t_paired_test(group1, group2, alpha=0.05):

    assert len(group1) == len(group2)

    t_stat, p_value = stats.ttest_rel(group1, group2)
    # print(f"t-statistic: {t_stat}, p-value: {p_value}")

    differences = np.array(group1) - np.array(group2)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    effect_size = mean_diff / std_diff

    n = len(group1)
    alpha = 0.05  # 显著性水平通常设为0.05
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, 
                                    nobs1=n, 
                                    alpha=alpha, 
                                    ratio=1, 
                                    alternative='two-sided')
    # print(f'power: {power:.3f}')
    return t_stat, p_value, power

def chi_test(likelihood_ratios, df, alpha=0.05):

    # chi-squared test critical value
    critical_value = chi2.ppf(1 - alpha, df)
    power_count = sum(lr > critical_value for lr in likelihood_ratios)
    return power_count / len(likelihood_ratios)

def compute_returns(policy, env, gamma=0.99, iterations=50):

    def compute_trajectory_return(trajectory, gamma=0.99):
        return sum(reward * (gamma ** i) for i, (_, _, reward, _) in enumerate(trajectory))

    returns = []
    for _ in range(iterations):
        obs, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            action = policy[obs]
            obs_, reward, done, _, info = env.step(action)
            trajectory.append((obs, action, reward, obs_))
            obs = obs_
        returns.append(compute_trajectory_return(trajectory, gamma=gamma))
    return returns

def main():
    '''
    Visualize a value table stored in a CSV file as a heatmap on top of a maze.

    Args:
        csv_path (str): Path to the CSV file containing the value table.
            The CSV file should have the same shape as the maze.
            The last column of the CSV file should contain the values to be visualized.
        save_path (str): Path to save the resulting visualization image.

    Raises:
        ValueError: If csv_path does not contain a valid map_id.

    Returns:
        None
    '''
    # read_path = 'scripts/save/model/qvalue/map_0_q_table_2000.csv'
    # save_path = '../scripts/save/img/test.png'
    
    # if "map_" in read_path:
    #     result = read_path.split("map_")[1][0]
    #     map_id = int(result)
    # else:
    #     raise ValueError("csv_path must contain map_id")

    # v_table = np.round(read_csv(read_path)[:, -1].reshape(arr_shape), decimals=2)

    # draw_maze(maze, v_table, save_path=save_path)
    # for i in range(7):
    maze = MAP_LIB[0]
    maze = np.array([list(map(int, binary_str)) for binary_str in maze])
    arr_shape = [len(maze), len(maze[0])]
    save_path = f'maze_{0}.pdf'
    policy_dir = f'map_0_policy_2000.csv'
    policy = read_csv(policy_dir)[:, -1]
    policy = policy.reshape(arr_shape)
    draw_maze(maze, values=policy, save_path=save_path)


if __name__ == "__main__":
    main()