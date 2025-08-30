# train stable-baseline3 type policy for agents
import hydra
from omegaconf import DictConfig
from stable_baselines3 import A2C, PPO

from testing.envs.maze.generator import genMaze, MAP_LIB
from testing.envs.carl.gymEnvGen import *


def train_maze_policy(config: DictConfig):
    # Initialize environment
    map = MAP_LIB[config.env.default_map_id]
    env = genMaze(
        map_size=tuple(config.env.map_size), 
        obstacle_locs=map, 
        max_prob=config.env.default_prob, 
        start_loc=tuple(config.env.start_loc), 
        goal_loc=tuple(config.env.goal_loc), 
        deterministic=config.env.deterministic
    )

    # Initialize agent
    agent = PPO(config.train.policy, env, verbose=config.train.verbose)

    # Train agent
    agent.learn(total_timesteps=config.train.total_timesteps)

    # Save agent
    agent.save(config.train.save_path)
    


def train_gym_policy(config: DictConfig):
    # Initialize environment
    env = gym.make(config.env.env_name)

    # Initialize agent
    agent = PPO("MlpPolicy", env, verbose=1)

    # Train agent
    agent.learn(total_timesteps=config.train.total_timesteps)

    # Save agent
    agent.save(config.train.save_path)


@hydra.main(version_base=None, config_path="../configs", config_name="maze")
def train_maze_policy_main(config: DictConfig):
    """Main function for training the base model"""
    train_maze_policy(config)
    
    
@hydra.main(version_base=None, config_path="../configs", config_name="gym")
def train_gym_policy_main(config: DictConfig):
    """Main function for training the base model"""
    train_gym_policy(config)
    

if __name__ == "__main__":
    train_gym_policy_main()