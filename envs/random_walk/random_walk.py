import numpy as np
from copy import deepcopy

class RandomWalkEnv():
    def __init__(self, num_states=5, max_steps=20):
        self.num_states = num_states
        self.state = self.num_states // 2
        self.max_steps = max_steps
        self.reset()

    def step(self, prob=0.7):
        coin_flip = np.random.choice(['heads', 'tails'], p=[prob, 1-prob])
        state = deepcopy(self.state)
        if coin_flip == 'heads':
            self.state += 1  # 向右移动
            delta = 1
        else:
            self.state -= 1  # 向左移动
            delta = -1

        self.steps += 1

        done = (self.state == 0 or self.state == self.num_states - 1) or (self.steps == self.max_steps)
        # reward = int(self.state == self.num_states - 1)

        return state, delta, done

    def reset(self):
        self.state = self.num_states // 2
        self.steps = 0
        return self.state
