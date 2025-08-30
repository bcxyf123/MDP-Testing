import os
import numpy as np
import torch
import types
import hydra
from omegaconf import DictConfig
from stable_baselines3 import A2C, PPO

from testing.detector.detector import MDPDetector
from testing.detector.sampler import sample

from testing.envs.carl.gymEnvGen import gen_gym


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


@hydra.main(version_base=None, config_path="../configs", config_name="gym")
def main(config: DictConfig):
    # Extract parameters from configuration
    
    env_config = config.env
    test_config = config.detect

    env_name = env_config.env_name
    param_name = env_config.param_name
    load_base_policy_path = env_config.load_base_policy_path
    
    detector_type = test_config.detector_type
    data_type = test_config.data_type
    
    num_episodes_default = test_config.num_episodes_default
    num_episodes_test = test_config.num_episodes_test

    sample_method = test_config.sample_method
    n_MC_samples = test_config.n_MC_samples

    # initialize environment using gen_gym function
    default_carl_env, _ = gen_gym(env_name=env_name, parameter=param_name)
    test_carl_env, _ = gen_gym(env_name=env_name, parameter=param_name)
    
    # set context_id
    if env_config.default_context_id is None:
        default_context_id = 0
    else:
        default_context_id = env_config.default_context_id
    default_carl_env.context_id = default_context_id
    if env_config.test_context_id is None:
        test_context_id = np.random.randint(len(test_carl_env.contexts))
    else:
        test_context_id = env_config.test_context_id
    test_carl_env.context_id = test_context_id

    default_carl_env.reset()
    test_carl_env.reset()
    
    # carl env needs unwrap
    default_env = default_carl_env.env
    test_env = test_carl_env.env

    # set base agent
    base_agent = PPO.load(load_base_policy_path, env=default_env)
    base_agent.compute_log_likelihood = types.MethodType(sb3_compute_log_likelihood, base_agent)

    detector = MDPDetector(detector_type=detector_type, data_type=data_type, base_agent=base_agent, gamma=test_config.gamma)
    
    # construct confidence interval and run detection
    reject_region = detector.dist_estimation(
        default_env=default_env, 
        sample_func=sample, 
        num_episodes_default=num_episodes_default, 
        num_episodes_test=num_episodes_test, 
        sample_method=sample_method, 
        n_samples=n_MC_samples
    )

    # run single test
    result = detector.run_test(
        reject_region=reject_region, 
        default_env=default_env, 
        test_env=test_env, 
        sample_func=sample, 
        num_episodes_default=num_episodes_default, 
        num_episodes_test=num_episodes_test
    )
    
    if result:
        print(f"Reject null hypothesis: environments are different")
    else:
        print(f"Fail to reject null hypothesis: environments are the same")


if __name__ == "__main__":
    
    main()
