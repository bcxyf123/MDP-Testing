import sys
sys.path.append('../..')
from envs.gym_randomgrid.generator import MAP_LIB
from utils.utils import *

if __name__ == '__main__':
    for i in range(10):
        maze = MAP_LIB[i]
        values = read_csv(f'save/model/policy/map_{i}_policy_1000.csv')[:, 1].reshape(4, 8)
        maze_int = np.array([list(map(int, binary_str)) for binary_str in maze])
        draw_maze(maze_int, values, end=None, save_path=f'maze_{i}')