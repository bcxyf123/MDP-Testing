# NOTE: to run this you must install additional dependencies
import logging, os, sys, time
sys.path.append("../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import gymnasium as gym
from gymnasium.utils.save_video import save_video

from algo.value_eval import mc_eval
from envs.gym_randomgrid.generator import genEnv
from utils.utils import *


def main(policy_id=0, map_id=0, read_path=None, save_path=None):
    if read_path is None:
        read_path = f'save/model/policy/map_{policy_id}_policy_2000.csv'
    if save_path is None:
        save_path = f'save/eval/map_{map_id}_v_table_{policy_id}.csv'
    policy = read_csv(read_path)[:, -1]
    env = genEnv(map_id=map_id, start_loc=(3,0), goal_loc=(0,7), deterministic=False)
    env.reset()
    v_s0, v_table, delta_list = mc_eval(policy, env, iterations=1000)

    plot(delta_list, )
    # save_csv(v_table, save_path)
    # arr_visualize(save_path, f'save/img/eval/map_{map_id}_v_table_{policy_id}.png')
    # print(env.to_xy(np.argmax(v_table)))
    print(v_s0)


if __name__=='__main__':

    # for i in range(7):
    #     main(policy_id=0, map_id=i)

    main(policy_id=0, map_id=0)
