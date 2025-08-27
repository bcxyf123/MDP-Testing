from __future__ import annotations
import sys
sys.path.append("..")
import numpy as np
from envs.gym_simplegrid.envs.simple_grid import SimpleGridEnv

class SimpleGridMDP(SimpleGridEnv):
    def __init__(self, 
                 obstacle_map: str | list[str] = None, 
                 deterministic: bool = True, 
                 max_prob: float = 0.6,
                 max_episode_steps: int = 100,
                 env_options: dict = None,
                 env_seed: int = None,
                 gamma: float = 0.9,
                 render_mode: str | None = None):
        super().__init__(obstacle_map, deterministic, max_prob, max_episode_steps, render_mode)
        self.n = self.nrow*self.ncol
        self.max_episode_steps = max_episode_steps
        self.discount_factor = gamma
        self.reset(env_seed, options=env_options)
    

    def get_transitions(self, state, action):
        '''
        Given a state and an action, returns a list of possible next states, the probability of each next state,
        and the reward associated with each next state.

        Args:
            state (int): The current state.
            action (int): The action to take.

        Returns:
            Tuple[List[int], List[float], List[float]]: A tuple containing a list of possible next states, the probability
            of each next state, and the reward associated with each next state.
        '''
        observations = []
        rewards = []

        self.agent_xy = self.to_xy(state)
        row, col = self.agent_xy

        actions, probs = self.get_transprob(row, col, action)

        for i, a in enumerate(actions):

            self.agent_xy = self.to_xy(state)
            dx, dy = self.MOVES[a]

            # Compute the target position of the agent
            target_row = row + dx
            target_col = col + dy

            # Compute the reward
            self.reward = self.get_reward(target_row, target_col)
            
            # Check if the move is valid
            if self.is_in_bounds(target_row, target_col):
                self.agent_xy = (target_row, target_col)

            observations.append(self.get_obs())
            rewards.append(self.reward)

        return observations, probs, rewards

    
    def get_discount_factor(self, ):
        return self.discount_factor
    
    def get_pfunc(self, state, action, next_state):
        '''
        Returns the probability of transitioning from the current state to the next state given the action taken.

        Args:
            state (int): The current state.
            action (int): The action taken.
            next_state (int): The next state.

        Returns:
            float: The probability of transitioning from the current state to the next state given the action taken.
        '''
        observations, probs, rewards = self.get_transitions(state, action)
        prob = 0
        if next_state in observations:
            indices = [i for i, x in enumerate(observations) if x == next_state]
            prob = sum([probs[i] for i in indices])
        if prob == 0:
            prob = 1e-6
        return prob
    

    def get_rfunc(self, state, action,):
        '''
        Returns the average reward for a given state and action.

        Args:
        - state: the current state of the MDP
        - action: the action taken in the current state

        Returns:
        - r: the average reward for the given state and action
        '''
        observations, probs, rewards = self.get_transitions(state, action) 
        r = np.multiply(probs, rewards).sum()

        return r