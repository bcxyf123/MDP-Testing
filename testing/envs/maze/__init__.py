
from gymnasium.envs.registration import register

register(
    id='SimpleGrid-v0',
    entry_point='testing.envs:SimpleGridEnv',
    max_episode_steps=100
)

register(
    id='SimpleGrid-8x8-v0',
    entry_point='testing.envs:SimpleGridEnv',
    max_episode_steps=100,
    kwargs={'obstacle_map': '8x8'},
)

register(
    id='SimpleGrid-4x4-v0',
    entry_point='testing.envs:SimpleGridEnv',
    max_episode_steps=100,
    kwargs={'obstacle_map': '4x4'},
)