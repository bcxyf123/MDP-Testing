import sys
sys.path.append('../..')

from envs.gym_simplegrid.generator import genMap

for i in range(1, 11):
    genMap(obstacle_num=i, num_maps=10, draw=True)