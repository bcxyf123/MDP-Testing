from gymnasium.envs.registration import register
from .covid_env import CovidEnv

register(
    id="CovidEnv-v0",
    entry_point="testing.envs.covid_env:CovidEnv",
)