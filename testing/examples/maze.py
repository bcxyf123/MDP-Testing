import os
import numpy as np
import torch
import types
import pandas as pd
import hydra
from omegaconf import DictConfig
from scipy.stats import rv_discrete
from stable_baselines3 import A2C, PPO

from testing.detector.detector import MDPDetector
from testing.detector.sampler import sample
from testing.detector.agent import CustomAgent

from testing.envs.maze.generator import genMaze, MAP_LIB

# add compute_log_likelihood function for stable-baseline3 type policy
def sb3_compute_log_likelihood(self, trans):
    
    obs, action, reward, next_obs = trans
    
    obs = np.expand_dims(obs, axis=0)

    obs_tensor = torch.tensor(obs).float().to(self.device)
    action_tensor = torch.tensor(action).to(self.device)
    
    with torch.no_grad():
        values, log_prob, entropy = self.policy.evaluate_actions(obs_tensor, action_tensor)
    log_prob = log_prob.detach().cpu().numpy()
    
    return log_prob


@hydra.main(version_base=None, config_path="../configs", config_name="maze")
def main(config: DictConfig):
    # Extract parameters from configuration
    
    env_config = config.env
    test_config = config.detect

    default_map_id = env_config.default_map_id
    test_map_id = env_config.test_map_id

    default_prob = env_config.default_prob
    test_prob = env_config.test_prob

    load_base_policy_path = test_config.load_base_policy_path

    detector_type = test_config.detector_type
    data_type = test_config.data_type

    num_episodes_default = test_config.num_episodes_default
    num_episodes_test = test_config.num_episodes_test

    sample_method = test_config.sample_method
    n_MC_samples = test_config.n_MC_samples

    default_map = MAP_LIB[default_map_id]
    test_map = MAP_LIB[test_map_id]
    
    # initialize environment

    default_env = genMaze(
        map_size=tuple(env_config.map_size), 
        obstacle_locs=default_map, 
        max_prob=default_prob, 
        start_loc=tuple(env_config.start_loc), 
        goal_loc=tuple(env_config.goal_loc), 
        deterministic=env_config.deterministic
    )
    test_env = genMaze(
        map_size=tuple(env_config.map_size), 
        obstacle_locs=test_map, 
        max_prob=test_prob, 
        start_loc=tuple(env_config.start_loc), 
        goal_loc=tuple(env_config.goal_loc), 
        deterministic=env_config.deterministic
    )

    # set base agent
    base_agent = PPO.load(load_base_policy_path, env=default_env)
    base_agent.compute_log_likelihood = types.MethodType(sb3_compute_log_likelihood, base_agent)

    # base_agent = CustomAgent(default_env.action_space, policy=lambda action_space=None, state=None: q_policy(action_space, state, config))

    detector = MDPDetector(detector_type=detector_type, data_type=data_type, base_agent=base_agent, gamma=test_config.gamma)
    reject_region = detector.dist_estimation(default_env=default_env, sample_func=sample, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test, sample_method=sample_method, n_samples=n_MC_samples)

    # run single test
    result = detector.run_test(reject_region=reject_region, default_env=default_env, test_env=test_env, sample_func=sample, num_episodes_default=num_episodes_default, num_episodes_test=num_episodes_test)
    
    if result:
        print("Reject null hypothesis: environments are different")
    else:
        print("Fail to reject null hypothesis: environments are the same")
    
    
if __name__ == "__main__":
    
    main()