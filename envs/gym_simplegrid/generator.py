import os
import sys
sys.path.append('..')
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from envs.gym_simplegrid.mdps.grid_mdp import SimpleGridMDP
from copy import deepcopy

# in case the randomly generated map doesn't have a valid path
# to ensure the state space is the same, the number of obstacles are the same
# 8x4 map
MAP_LIB = [
    # default
    ['00001000', 
     '00100010', 
     '00000010', 
     '00010000'], 

    # similar to default
    ['00001000', 
     '00100011', 
     '00000011', 
     '00010000'], 

     
    # no obstacles
    ['00000000',
     '00000000',
     '00000000',
     '00000000',],
 
    # different from 
    ['00110000', 
     '00110001', 
     '10000100', 
     '00010010'], 

    ['00011000', 
     '10010000', 
     '00010101', 
     '01000000'], 
     
    ['10000100', 
     '10010001', 
     '00100001', 
     '01100001'], 

    ['10110010', 
     '01000010', 
     '01011010', 
     '00001000'],

]

# MAP_LIB = [

#     # 5
#     ['00000100', '00010000', '11000001', '00000000'],['10000000', '00011000', '00000000', '00010000'],['00100000', '00000010', '00001001', '00000100'],['00000010', '10000100', '11000000', '00000000'],['00100000', '00010101', '00000000', '00100000'],
#     # 6
#     ['00000010', '00000000', '00110000', '00000111'],['00001010', '10000100', '00000100', '00100000'],['10000000', '00000010', '01000100', '00001001'],['00010000', '10010001', '00000000', '00000100'],['01000000', '00110000', '00000000', '00101100'],
#     # 7
#     ['00010100', '00000101', '11000000', '00000100'],['01000000', '00111100', '00100100', '00000000'],['00100000', '00010010', '01000010', '00001100'],['10000100', '11100010', '00010000', '00000000'],['00000000', '00000011', '00001000', '00101000'],
#     # 8
#     ['00010100', '01001000', '10000010', '00110000'],['10000100', '01000000', '00110010', '00000110'],['00010000', '00000011', '00000100', '01000011'],['00110010', '00000000', '10101000', '00000100'],['00101100', '10001000', '01000000', '00000100'],
#     # 9
#     ['10000000', '10000110', '00010100', '00011000'],['01101100', '00100001', '01000100', '00000100'],['01100000', '00001010', '01000010', '01000110'],['00000110', '00010000', '01000000', '00011101'],['10011000', '00110000', '10100000', '00000010'],
#     # 10
#     ['11000000', '00000010', '01010001', '00011100'],['10100110', '10000000', '00000000', '00000011'],['01000100', '00000000', '01000100', '00100001'],['10001000', '01000010', '01000000', '00001010'],['00100110', '00000000', '00010011', '01001110'],
#     # 11
#     ['11000100', '00011010', '01100000', '00000010'],['01000000', '11011001', '00100000', '00000010'],['01000010', '00110000', '00000010', '00111011'],['00000000', '10010001', '00100100', '00001010'],['11000110', '01101000', '00100001', '00001100'],
#     # 12
#     ['10001100', '00000001', '10000100', '00110001'],['00110000', '10001101', '00000000', '01100001'],['10011000', '11001011', '00000001', '00000110'],['11010100', '11000000', '00011001', '00010010'],['11010000', '01000000', '00011011', '00001011'],
#     # 13
#     ['00110010', '10100000', '10000011', '00001000'],['11110000', '11000000', '11010110', '00010000'],['00011100', '00110101', '00000100', '01010000'],['00100000', '10001011', '10001010', '00110010'],['01010000', '01010001', '00100011', '00001001'],
#     # 14
#     ['01011000', '11011011', '00000000', '01001100'],['00111000', '10110010', '10000101', '00001010'],['01010100', '11001001', '00000010', '00111110'],['00111000', '00000000', '10101111', '00010011'],['10011100', '10001000', '00000100', '01000001'],
#     # 15
#     ['10111000', '00000001', '00100001', '00011110'],['01011100', '11000101', '10000000', '00001000'],['11100000', '00000010', '10111000', '00010001'],['11110000', '00100110', '00101011', '00000101'],['10011110', '11100100', '00000000', '00100011'],
#     # 16
#     ['00011110', '00000000', '01000011', '01011111'],['00010000', '01000010', '10010000', '00010111'],['11000000', '00100101', '01010101', '00000001'],['01110000', '10101100', '11000001', '00011011'],['00000100', '11110001', '01000110', '00010011'],

# ]


# probs_lib = [[max_prob, (1-max_prob)/2, (1-max_prob)/2] for max_prob in np.arange(0.9, 0.5, -0.01)]
# probs_lib = [[max_prob, (1-max_prob)/2, (1-max_prob)/2] for max_prob in np.arange(0.5, 1.0, 0.01)]
# probs_lib = [[0.6, 0.1, 0.1]]

def genMap(map_size=(4, 8), obstacle_num=7, num_maps=10, draw=False):

    def draw_maze(maze, start=(3, 0), end=(0, 7), save_path=None):
        cmap = plt.cm.binary
        norm = plt.Normalize(0, 1)

        fig, ax = plt.subplots()
        ax.imshow(maze, cmap=cmap, norm=norm, interpolation='none', alpha=0.5)

        xticks = [x-0.5 for x in range(maze.shape[1])]
        yticks = [y-0.5 for y in range(maze.shape[0])]
        ax.set_xticks(xticks, minor=False)
        ax.set_yticks(yticks, minor=False)
        ax.grid(which="major", color="black", linestyle='-', linewidth=2)

        # start and goal points
        if start is not None:
            rect_start = Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, linewidth=2, facecolor='red', alpha=0.5)
            ax.add_patch(rect_start)
        if end is not None:
            rect_end = Rectangle((end[1]-0.5, end[0]-0.5), 1, 1, linewidth=2, facecolor='green', alpha=0.5)
            ax.add_patch(rect_end)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.savefig(save_path)

    validmap_list = []
    while len(validmap_list) < num_maps:
        layout = np.zeros(map_size)
        indices = [(np.random.choice(range(map_size[0])), np.random.choice(range(map_size[1]))) for _ in range(obstacle_num)]
        for index in indices:
            layout[index[0]][index[1]] = 1
        maze = deepcopy(layout)
        if is_path_exists(maze, (3,0), (0,7)):
            validmap_list.append(layout)
            if draw:
                save_dir = f'maps/{obstacle_num}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                draw_maze(layout, save_path=os.path.join(save_dir, f'map_{len(validmap_list)}.png'))
        map_lib = [arr2str(validmap) for validmap in validmap_list]
    return map_lib

def arr2str(arr):
    return ["".join(str(int(x)) for x in row) for row in arr]

def is_path_exists(maze, start, end):

    rows = maze.shape[0]
    cols = maze.shape[1]

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols or maze[row][col] == 1:
            return False
        
        if row == end[0] and col == end[1]:
            return True
        
        maze[row][col] = 1

        if dfs(row - 1, col):  # 上
            return True
        if dfs(row + 1, col):  # 下
            return True
        if dfs(row, col - 1):  # 左
            return True
        if dfs(row, col + 1):  # 右
            return True
        
        # maze[row][col] = 0
        
        return False

    return dfs(start[0], start[1])

def genEnv(map_size=None, obstacle_locs=None, map_id=None, max_prob=0.6, start_loc=None, goal_loc=None, deterministic=False, seed=None):
    # in non-approximation methods, it's best to keep the state space the same
    if map_size is None:
        map_size = np.random.randint(5, size=(1, 2))
    if map_id is None:
        map_id = np.random.randint(0, len(MAP_LIB))
    obstacle_locs = MAP_LIB[map_id]
    if start_loc is None:
        start_loc = np.random.randint(1, [map_size[0], map_size[1]])
        while(obstacle_locs[start_loc[0]][start_loc[1]]):
            start_loc = np.random.randint(1, [map_size[0], map_size[1]])
    if goal_loc is None:
        goal_loc = np.random.randint(1, [map_size[0], map_size[1]])
        while(obstacle_locs[goal_loc[0]][goal_loc[1]] or start_loc == goal_loc):
            goal_loc = np.random.randint(1, [map_size[0], map_size[1]])
    if seed is None:
        seed = np.random.randint(0, 10000)
    options = {'start_loc': start_loc, 'goal_loc': goal_loc}

    env = gym.make(
        'SimpleGrid-v0',
        max_episode_steps=100,
        obstacle_map=obstacle_locs, 
        deterministic=deterministic,
        max_prob=max_prob,
        render_mode='human',
    )
    env.reset(seed=seed, options=options)

    return env


def genMDP(map_size=None, obstacle_locs=None, map_id=None, max_prob=0.6, start_loc=None, goal_loc=None, deterministic=False, seed=None):
    # in non-approximation methods, it's best to keep the state space the same
    if map_size is None:
        map_size = np.random.randint(5, size=(1, 2))
    if obstacle_locs is None:
        if map_id is None:
            map_id = np.random.randint(0, len(MAP_LIB))
    obstacle_locs = MAP_LIB[map_id]
    if start_loc is None:
        start_loc = np.random.randint(1, [map_size[0], map_size[1]])
        while(obstacle_locs[start_loc[0]][start_loc[1]]):
            start_loc = np.random.randint(1, [map_size[0], map_size[1]])
    if goal_loc is None:
        goal_loc = np.random.randint(1, [map_size[0], map_size[1]])
        while(obstacle_locs[goal_loc[0]][goal_loc[1]] or start_loc == goal_loc):
            goal_loc = np.random.randint(1, [map_size[0], map_size[1]])
    if seed is None:
        seed = np.random.randint(0, 10000)
    options = {'start_loc': start_loc, 'goal_loc': goal_loc}

    mdp = SimpleGridMDP(
        obstacle_map=obstacle_locs, 
        deterministic=deterministic,
        max_prob=max_prob,
        max_episode_steps=100,
        env_options=options,
        env_seed=seed,
    )
    mdp.reset(seed=seed, options=options)

    return mdp