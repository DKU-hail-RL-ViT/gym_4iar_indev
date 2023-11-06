import random
import numpy as np

from collections import defaultdict, deque

import torch

# Four in a row task
from gym_4iar.model.qrdqn import QRDQN
import gymnasium as gym


if __name__ == '__main__':

    env = gym.make("CartPole-v1", render_mode="human")

    policy_kwargs = dict(n_quantiles=50)
    model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_starts=1000)
    model.learn(total_timesteps=100_000, log_interval=4, progress_bar=True)
    model.save("qrdqn_cartpole")

    del model  # remove to demonstrate saving and loading

    model = QRDQN.load("qrdqn_cartpole")

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
