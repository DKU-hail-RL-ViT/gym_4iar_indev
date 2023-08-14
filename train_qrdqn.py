import os
import yaml
import argparse
from datetime import datetime

from gym_4iar.agent import QRDQNAgent

# Four in a row task
import numpy as np
import matplotlib.pyplot as plt
import fiar_env
import numpy as np


def embedding_function(N):
    # Define your embedding function here
    # For simplicity, let's assume it's a linear function for now
    return N


def planning_algorithm(Kmax, Kmin, theta, R, M, T):
    for episode in range(1, M + 1):
        for t in range(1, T + 1):
            dsim = 1
            Rrem = R
            P = 1
            j = t

            while Rrem > 0:
                N = 2 ** P
                K = embedding_function(N)

                if abs(K) < Kmin:
                    K = np.sign(K) * Kmin
                elif abs(K) > Kmax:
                    K = np.sign(K) * Kmax

                Q_sj_aj = np.mean(Z_sj_aj_samples)  # Z(sj, aj) samples

                # Execute actions, update networks, and other steps
                # You would need to provide implementations for these parts

                if Q_sj_b1 - Q_sj_b2 > theta:
                    # Execute action aj = b1 and update variables
                    # Update networks using L1, increment j and dsim
                    pass
                else:
                    P += 1

                Rrem -= K * Na  # Deduct remaining resources

            # Compute estimated and target values, compute loss L1, L2
            # Minimize L1 + L2

        # End of inner loop (for t)
    # End of outer loop (for episode)



def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create the agent and run.
    agent = QRDQNAgent(
        env=env, cuda=args.cuda, **config)
    agent.run()




if __name__ == '__main__':

    # Define other necessary functions and variables as needed

    # Example usage
    Kmax = 100
    Kmin = 1
    theta = 0.1
    R = 10
    M = 5
    T = 100
    planning_algorithm(Kmax, Kmin, theta, R, M, T)


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
            action2d = (action2d[0][0], action2d[1][0])
            if map_taken[action2d] != -1:
                break

        env.player = 1
        map_taken[action2d] = -1
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

    args = parser.parse_args()
    run(args)

