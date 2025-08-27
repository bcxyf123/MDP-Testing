# 导入需要使用的库,其中numpy是支持数组和矩阵运算的科学计算库,而matplotlib是绘图库
import numpy as np
import matplotlib.pyplot as plt


class Spec:
    def __init__(self, id):
        self.id = id

class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K=5, p=0.5):
        # self.probs = np.random.uniform(size=K, high=max_prob)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        self.probs = np.array([0.1, p])
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.best_rew = 1*self.best_prob  # 最大的获奖概率对应的奖励
        self.K = K
        self.spec = Spec(id='bandit')

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

