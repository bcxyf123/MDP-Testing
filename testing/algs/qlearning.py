import numpy as np

import logging, os, sys
from envs_old.gym_simplegrid.envs import SimpleGridEnv
from datetime import datetime as dt
import gymnasium as gym
from gymnasium.utils.save_video import save_video


class QLearner:
    def __init__(self, num_states, num_actions, lr=0.2, gamma=0.9, epsilon=0.9, exp_decay=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = exp_decay
        self.cur_policy = np.random.randint(num_actions, size=num_states)
        self.q_table = np.zeros((num_states, num_actions))

    def update(self, s, a, s_prime, r):
        q_prime = np.max(self.q_table[s_prime])
        old_q_value = self.q_table[s, a]
        learned_value = r + self.gamma * q_prime - old_q_value
        self.q_table[s, a] += self.alpha * learned_value
        self.cur_policy[s] = np.argmax(self.q_table[s])

    def act(self, s):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return self.cur_policy[s]
        
    def eval_act(self, s):
        return self.cur_policy[s]

    def update_episode(self):
        self.epsilon *= self.decay_rate