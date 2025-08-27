import numpy as np
import pprint
import sys
from envs.gym_simplegrid.mdps.grid_mdp import SimpleGridMDP
from envs.gym_simplegrid.generator import MAP_LIB
from utils.utils import *
from algo.dp_eval import policy_eval

# def policy_eval(policy, mdp, discount=0.9,theta=0.00001):
#     v_table = np.zeros(mdp.observation_space.n)
#     delta_list = []
#     while True:
#         delta = 0
#         for s in range(mdp.observation_space.n):
#             v = 0
#             a = policy[s]
#             next_states, probs, rewards = mdp.get_transitions(s, a)
#             for s_, prob, r in zip(next_states, probs, rewards):
#                 v += prob*(r + discount* v_table[s_])
#             delta=max(delta, np.abs(v-v_table[s]))
#             v_table[s]=v
#         delta_list.append(delta)
#         if delta < theta:
#             break
#     mdp.close()

#     return v_table, delta_list


def main(policy_id=0, map_id=0):
    read_path = f'save/model/policy/map_{policy_id}_policy_2000.csv'
    save_path = f'save/eval/dp/map_{map_id}_v_table_{policy_id}.csv'
    policy = read_csv(read_path)[:, -1]
    mdp = SimpleGridMDP(obstacle_map=MAP_LIB[map_id], deterministic=False, gamma=0.9)
    v_table, delta_list = policy_eval(policy, mdp, discount=0.9, theta=0.00001)
    save_csv(v_table, save_path)
    plot(delta_list, f'C:/Users/taizun/Desktop/ood/OOD/scripts/save/img/dp/delta_{map_id}_{policy_id}.png', xlabel='iteration', ylabel='delta')
    draw_maze(save_path, f'C:/Users/taizun/Desktop/ood/OOD/scripts/save/img/dp/map_{map_id}_v_table_{policy_id}.png')
    print(mdp.to_xy(np.argmax(v_table)))


if __name__ == '__main__':
    main(0, 0)