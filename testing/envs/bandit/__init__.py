from gymnasium.envs.registration import register

register(
    id='bandit-v0',
    entry_point='testing.envs:BanditEnv',
    max_episode_steps=200
)
