import random
import numpy as np

def generate_coordinates():
    # 生成第一组坐标
    x1 = random.randint(0, 3)
    y1 = random.randint(0, 7)
    coord1 = (x1, y1)

    # 生成第二组坐标，确保它们不重合
    while True:
        x2 = random.randint(0, 3)
        y2 = random.randint(0, 7)
        coord2 = (x2, y2)
        if coord2 != coord1:
            break

    return coord1, coord2


def coord_minus_abs(coord1, coord2):
    return [abs(x-y) for x,y in zip(coord1, coord2)]

def compute_prob(coords: list, start_loc=(0,3), max_prob=0.7):
    grid_prob = max_prob/10
    probs = [grid_prob*sum(coord_minus_abs(coord, start_loc)) for coord in coords]

    return coords, probs

def draw_maze(coords):
    maze = np.zeros((4, 8), dtype=int)
    for coord in coords:
        maze[coord[0]][coord[1]] = 1
    str_maze = [''.join(map(str, row)) for row in maze]

    return str_maze

# 测试代码
np.random.seed(1)
for i in range(10):
    coord1, coord2 = generate_coordinates()
    # print(coord1, coord2)
    print(draw_maze([coord1, coord2]))

    # coords, probs = compute_prob([coord1, coord2], start_loc=(0,3), max_prob=0.7)
    # print(coords, probs)