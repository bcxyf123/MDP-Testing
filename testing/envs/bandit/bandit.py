# import numpy as np
# import matplotlib.pyplot as plt


# class Spec:
#     def __init__(self, id):
#         self.id = id

# class BernoulliBandit:
#     """ 伯努利多臂老虎机,输入K表示拉杆个数 """
#     def __init__(self, K=5, p=0.5):
#         # self.probs = np.random.uniform(size=K, high=max_prob)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
#         self.probs = np.array([0.1, 0.2, 0.3, 0.4, p])
#         # 概率
#         self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
#         self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
#         self.best_rew = 1*self.best_prob  # 最大的获奖概率对应的奖励
#         self.K = K
#         self.spec = Spec(id='bandit')

#     def step(self, k):
#         # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
#         # 获奖）
#         if np.random.rand() < self.probs[k]:
#             return 1
#         else:
#             return 0



import numpy as np
import random
from gymnasium import spaces, Env, core

class BanditEnv(Env):
    def __init__(self, num_arms=5, bandit_type='Bernoulli', max_steps=10, **kwargs):
        self.num_arms = num_arms
        self.bandit_type = bandit_type                  # choose from ['Bernoulli', 'Gaussian'] 
        self.action_space = spaces.Discrete(num_arms)   # which arm to pull in each step
        self.observation_space = spaces.Discrete(1)     # single observation space in bandit
        self.max_steps=max_steps
        self.steps = 0

        self.kwargs = kwargs

    def step(self, action):
        if self.bandit_type == 'Bernoulli':
            p = self.kwargs.get('p', 0.5)
            if p is None:
                raise ValueError("Bernoulli bandit requires a list of probabilities")
            reward = np.random.binomial(1, p[action])
        elif self.bandit_type == 'Gaussian':
            mu = self.kwargs.get('mu', 0)
            sigma = self.kwargs.get('sigma', 1)
            if mu is None or sigma is None:
                raise ValueError("Gaussian bandit requires a list of means and standard deviations")
            reward = np.random.normal(mu[action], sigma[action])
        done = True if self.steps >= self.max_steps-1 else False
        info = {"bandit type":self.bandit_type, "number of arms":self.num_arms, self.bandit_type + " parameters":self.kwargs, "steps":self.steps}
        self.steps += 1
        return 0, reward, done, False, info

    def reset(self, 
              seed: int | None = None, 
              options: dict | None = None
            ):
        # Set seed
        super().reset(seed=seed)
        
        if seed is not None:
            seed = random.randint(0, 10000)
        random.seed(seed)
        
        self.steps = 0
        
        return 0, {}
 
    def compute_log_likelihood(self, 
                           transition: list, # [s, a, r, s_]
                        ):
        """
        Compute the likelihood of a given (action, reward) tuple in the current environment.
        :param action: The selected arm (action)
        :param reward: The received reward
        :return: likelihood of the (action, reward) tuple
        """
        s, a, r, s_ = transition
        if self.bandit_type == 'Bernoulli':
            p = self.kwargs.get('p', 0.5)  # probabilities of the arms
            if p is None:
                raise ValueError("Bernoulli bandit requires a list of probabilities")
            # Bernoulli likelihood: if reward == 1, it's p[action], if reward == 0, it's 1 - p[action]
            likelihood = p[a] if r == 1 else (1 - p[a])

        elif self.bandit_type == 'Gaussian':
            mu = self.kwargs.get('mu', 0)
            sigma = self.kwargs.get('sigma', 1)
            if mu is None or sigma is None:
                raise ValueError("Gaussian bandit requires a list of means and standard deviations")
            # Gaussian likelihood using the probability density function (PDF)
            likelihood = (1 / (sigma[a] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - mu[a]) ** 2) / (sigma[a] ** 2))

        return np.log(likelihood)