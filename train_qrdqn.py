import os
import yaml
import argparse
from datetime import datetime

import wandb
# Four in a row task
import fiar_env
import torch

from gym_4iar.agent import QRDQNAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create the agent and run.
    agent = QRDQNAgent(
        env=env, cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':

    wandb.init(project="4iar_QR-DQN",
               entity='hails')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    parser.add_argument('--env_id', type=str, default='gym_4iar_env')
    parser.add_argument('--cuda', action='store_true')

    num_trials = 1
    env = fiar_env.Fiar()
    done = False

    args = parser.parse_args()
    run(args)

