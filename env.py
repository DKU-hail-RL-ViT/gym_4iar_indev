# NOTE: this code was mainly taken from:
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/atari_wrappers.py
from collections import deque

import numpy as np
import gym
from gym import spaces, wrappers
import cv2
cv2.ocl.setUseOpenCL(False)

import fiar_env

def make_atari(env_id):
    """
    Create a wrapped atari envrionment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = fiar_env.Fiar()
    return env

