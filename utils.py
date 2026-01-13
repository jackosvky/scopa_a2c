import gymnasium as gym
from .env import ScoponeEnv


def make_env():
    return ScoponeEnv()
