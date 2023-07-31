import os
import yaml
import argparse
from datetime import datetime

from gym_4iar.agent import QRDQNAgent

# Four in a row task
import numpy as np
import matplotlib.pyplot as plt
import fiar_env

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = fiar_env.Fiar()

    # Create the agent and run.
    agent = QRDQNAgent(
        env=env, cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    parser.add_argument('--env_id', type=str, default='gym_fiar_env')
    parser.add_argument('--cuda', action='store_true')
    num_trials = 1
    env = fiar_env.Fiar()

    done = False

    map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9).reshape(9, 4))
    map_taken = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9).reshape(9, 4))
    map_1d = np.int16(np.linspace(0,4*9-1, 4*9)).tolist()

    while not done:
        action = env.render(mode="terminal")
        while True:
            action = np.random.randint(len(map_1d))
            action2d = np.where(map==action)
            action2d = (action2d[0][0],action2d[1][0])
            if map_taken[action2d] != -1:
                break
        map_taken[action2d] = -1
        env.player = 0
        state, reward, done, info = env.step(action2d)

        if env.game_ended():
            env.render(mode="terminal")
            print("*" * 20)
            print('Game ended! [' + ('WHITE' if np.all(env.state_[2] == 1) else 'BLACK') + '] won!')
            SWITCHERS, STRRS = fiar_env.fiar_check(env.state_, loc=True)
            print("*" * 20)
            print('winning streak: ', STRRS)
            print('\n')
            break
        action = env.render(mode="terminal")

        while True:
            action = np.random.randint(len(map_1d))
            action2d = np.where(map==action)
            action2d = (action2d[0][0],action2d[1][0])
            if map_taken[action2d] != -1:
                break

        env.player = 1
        map_taken[action2d] = -1
        state, reward, done, info = env.step(action2d)

        if env.game_ended():
            env.render(mode="terminal")
            print("*" * 20)
            print('Game ended! [' + ('WHITE' if np.all(env.state_[2] == 1) else 'BLACK') + '] won!')
            SWITCHERS, STRRS = fiar_env.fiar_check(env.state_,loc=True)
            print("*" * 20)
            print('winning streak: ', STRRS)
            print('\n')
            break

    args = parser.parse_args()
    run(args)