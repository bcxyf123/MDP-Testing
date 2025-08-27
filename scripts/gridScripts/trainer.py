# NOTE: to run this you must install additional dependencies
import logging, os, sys, time
sys.path.append("../..")

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from algo.qlearning import QLearner
from envs.gym_simplegrid.generator import genEnv

# plt.rcParams['font.family'] = 'desired-font-family'


def save_csv(data, filepath):
    # with open(filepath, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in data:
    #         writer.writerow(row)
    # print(f"data saved in {filepath}")
    df = pd.DataFrame(data)
    df.to_csv(filepath)

class gridTrainer(object):
    def __init__(self, map_id) -> None:

    #     obstacle_map = [
    #     "10001000",
    #     "10010000",
    #     "00000001",
    #     "01000001",
    # ]
    #     self.env = gym.make(
    #     'SimpleGrid-v0', 
    #     obstacle_map=obstacle_map, 
    #     render_mode='human'
    # )
    #     self.env.reset()

        start_loc = (3, 0)
        goal_loc = (0, 7)

        self.map_id = map_id
        self.env = genEnv(map_id=map_id, start_loc=start_loc, goal_loc=goal_loc, max_prob=1.0, deterministic=False, seed=222333)
        # self.reset = envGen.reset

        num_states = self.env.observation_space.n
        num_actions = 4
        self.learner = QLearner(num_states, num_actions, gamma=0.99)

    # def train_one_epoch(self,):
    #     episode_rew = 0
    #     obs = self.reset()
    #     while(True):
    #         action = self.learner.act(obs)
    #         obs_, reward, done, _, info = self.env.step(action)
    #         episode_rew += reward
    #         self.learner.update(obs, action, obs_, reward)
    #         if done:
    #             break
    #         else:
    #             obs = obs_
    #     return episode_rew

    def train(self, epochs):
        total_reward = 0
        rewards = []
        for i in range(epochs):
            episode_rew = 0
            obs, _ = self.env.reset()
            print(f'Training epoch {i+1}')
            while(True):
                action = self.learner.act(obs)
                obs_, reward, done, _, info = self.env.step(action)
                episode_rew += reward
                self.learner.update(obs, action, obs_, reward)
                if done:
                    break
                else:
                    obs = obs_
            self.learner.update_episode()
            rewards.append(episode_rew)
            total_reward += episode_rew
        self.reward = total_reward / epochs
        self.env.close()
        save_csv(self.learner.cur_policy, filepath=f'save/model/policy/map_{self.map_id}_policy_{epochs}.csv')
        save_csv(self.learner.q_table, filepath=f'save/model/qvalue/map_{self.map_id}_q_table_{epochs}.csv')
        self.plot(epochs, rewards)
    
    def plot(self, epochs, rewards):
        episodes = range(epochs)
        plt.clf()
        plt.plot(episodes, rewards, marker='o', linestyle='-')

        plt.xlabel('episodes')
        plt.ylabel('rewards')
        
        plt.savefig(f'save/img/train/map_{self.map_id}_rewards_{epochs}.pdf', dpi=300)


if __name__=='__main__':

    trainer = gridTrainer(0)
    trainer.train(2000)

    # for i in range(5):
    #     trainer = gridTrainer(i)
    #     trainer.train(3000)

    # # Folder name for the simulation
    # FOLDER_NAME = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    # os.makedirs(f"log/{FOLDER_NAME}")

    # # Logger to have feedback on the console and on a file
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[logging.StreamHandler(sys.stdout)]
    # )
    # logger = logging.getLogger(__name__)

    # logger.info("-------------START-------------")

    # options ={
    #     'start_loc': 12,
    #     # goal_loc is not specified, so it will be randomly sampled
    # }

    # obstacle_map = [
    #     "10001000",
    #     "10010000",
    #     "00000001",
    #     "01000001",
    # ]
    
    # env = gym.make(
    #     'SimpleGrid-v0', 
    #     obstacle_map=obstacle_map, 
    #     render_mode='human'
    # )

    # obs, info = env.reset(seed=1, options=options)
    # rew = env.unwrapped.reward
    # done = env.unwrapped.done

    # logger.info("Running action-perception loop...")
    
    # with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
    #     f.write(f"step,x,y,reward,done,action\n")
        
    #     for t in range(500):
    #         #img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
            
    #         action = env.action_space.sample()
    #         f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")
            
    #         if done:
    #             logger.info(f"...agent is done at time step {t}")
    #             break
            
    #         obs, rew, done, _, info = env.step(action)
            
    # env.close()
    # if env.render_mode == 'rgb_array_list':
    #     frames = env.render()
    #     save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)
    # logger.info("...done")
    # logger.info("-------------END-------------")