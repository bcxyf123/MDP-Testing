# import gymnasium as gym
# from gymnasium import Env
import numpy as np
from typing import Dict
from .agent import Agent, CustomAgent
from stable_baselines3.common.base_class import BaseAlgorithm

# A typical sample function for gym environments
def sample(
    env,
    agent: Agent,
    gamma: float=None,
    num_episodes: int=10
    ) -> Dict:
    
    
    if env is None:
        raise ValueError("Environment for sampling is not provided.")
    if agent is None:
        raise ValueError("Agent for sampling is not provided.")
        
    # collect trajectories
    returns = []
    trajs = []
    # print("\n=== ENTER sample ===\n")

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        traj = []
        ret = 0
        
        if isinstance(agent, BaseAlgorithm):
            obs = env.reset()
            vec_env = agent.get_env()
            obs = vec_env.reset()
            obs = obs[0]
        
        while not (done or truncated):
            action = agent.predict(obs, deterministic=False)
            if isinstance(agent, BaseAlgorithm):
                action = np.squeeze(action[0]).item()
            obs_, reward, done, truncated, info = env.step(action)
            if "bandit" in env.spec.id:
                obs = info['steps']
            # traj.append([obs, action, reward, obs_])
            traj.append((obs, action, reward, obs_))
            if gamma is None:
                raise ValueError("discount factor is not assigned.")
            ret += gamma*reward
            obs = obs_
        returns.append(ret)
        trajs.append(traj)
        
    sample_dict = {"returns": returns, "trajs": trajs}
    
    return sample_dict