import numpy as np  # linear algebra
import matplotlib.pyplot as plt

from fiar_env import Fiar
from collections import defaultdict, deque
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_elo', type=int, default=1500)  # initial Elo rating.
parser.add_argument('--k_factor', type=int, default=32)  # sensitivity of the rating adjustment.
# parser.add_argument('--training_iterations', type=int, default=100)  # number of training iterations.
# parser.add_argument('--self_play_sizes', type=int, default=100)  # number of self-play games.
parser.add_argument('--c_puct', type=int, default=5)

args = parser.parse_args()

# Elo rating system
init_elo = args.init_elo
k_factor = args.k_factor

c_puct = args.c_puct


def policy_evaluate(env, player_1, player_2, n_games=100):
    result = 0
    turns = 0
    win_cnt = defaultdict(int)

    for i in range(n_games):
        winner = start_play(env, player_1, player_2)

        """ black standard win = 1, lose = 0, draw = 0.5 """

        if turns == 0 and winner == 1:  # p1 = black & p1 wins
            result = 1
        elif turns == 0 and winner == -0.5:  # p1 = black & p2 wins
            result = 0
        elif turns == 0 and winner == -1:  # p1 = black & draw
            result = 0.5

        elif turns == 1 and winner == -0.5:  # p1 = white & p1 wins
            result = 0
        elif turns == 1 and winner == 1:  # p1 = white & p2 wins
            result = 0.5
        elif turns == 1 and winner == -1:  # p1 = white & draw
            result = 1
        else:
            assert False  # should not reach here
        win_cnt[result] += 1

        # switch the player side
        player_1, player_2 = player_2, player_1
        turns = 1 - turns

    player_1, player_2 = player_2, player_1
    win_ratio = 1.0 * win_cnt[1] / n_games
    print("Win ratio:", 1.0 * win_cnt[1])
    return win_ratio


def start_play(env, player1, player2):
    """start a game between two players"""
    obs, _ = env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    move = None

    player_in_turn = players[current_player]

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, temp=0.1, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            obs, _ = env.reset()
            return winner


def elo_rating(p1, p2, result, turns, k=20):
    """
    calculate elo expected win rate, init expected win rate == 0.5
    """
    expected_rate = round(1.0 / (1.0 + 10 ** ((p1 - p2) / 400)), 3)
    opponent_rate = round(1 - expected_rate, 3)

    print("Expected rate:", expected_rate)
    print("Opponent rate:", opponent_rate)

    # Elo rating formula
    p1_rating = round(p1 + k * (result - expected_rate), 3)
    p2_rating = round(p2 + k * (opponent_rate - result), 3)

    return p1_rating, p2_rating


if __name__ == '__main__':
    env = Fiar()
    obs, _ = env.reset()
    win_rate = 0

    # player 1 info
    p1_rl_model = "AC"
    p1_n_playout = 10
    p1_quantiles = 32
    p1_elo = init_elo  # 1500

    # player 2 info
    p2_rl_model = "AC"
    p2_n_playout = 10
    p2_quantiles = 32
    p2_elo = init_elo  # 1500

    p1 = f"RL_{p1_rl_model}_nmcts{p1_n_playout}/train_088.pth"  # player 1
    p2 = f"RL_{p2_rl_model}_nmcts{p2_n_playout}/train_009.pth"  # player 2

    if p1_rl_model == "AC":
        p1_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                model_file=p1, rl_model=p1_rl_model)
    else:
        p1_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                p1_quantiles, model_file=p1, rl_model=p1_rl_model)

    if p2_rl_model == "AC":
        p2_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                model_file=p2, rl_model=p2_rl_model)
    else:
        p2_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                p2_quantiles, model_file=p2, rl_model=p2_rl_model)


    player_1 = MCTSPlayer(p1_net.policy_value_fn, c_puct, p1_n_playout, is_selfplay=0)
    player_2 = MCTSPlayer(p2_net.policy_value_fn, c_puct, p2_n_playout, is_selfplay=0)

    try:
        n_games = 50
        turns = 0
        win_cnt = defaultdict(int)

        # for 100 games to update the Elo rating
        for i in range(n_games):

            winner = start_play(env, player_1, player_2)

            """ black standard win = 1, lose = 0, draw = 0.5 """

            if turns == 0 and winner == 1:  # p1 = black & p1 wins
                result = 1
            elif turns == 0 and winner == -0.5:  # p1 = black & p2 wins
                result = 0
            elif turns == 0 and winner == -1:  # p1 = black & draw
                result = 0.5

            elif turns == 1 and winner == -0.5:  # p1 = white & p2 wins
                result = 1
            elif turns == 1 and winner == 1:  # p1 = white & p1 wins
                result = 0
            elif turns == 1 and winner == -1:  # p1 = white & draw
                result = 0.5
            else:
                assert False  # should not reach here

            print("Result:", result)

            if turns == 0:
                p1_elo, p2_elo = elo_rating(p1_elo, p2_elo, result, turns)
                print("Player A Elo:", p1_elo)
                print("Player B Elo:", p2_elo)
                print(f"{i + 1} / {n_games} \n")
            else:
                p2_elo, p1_elo = elo_rating(p2_elo, p1_elo, result, turns)
                p1_elo, p2_elo = p2_elo, p1_elo
                print("Player A Elo:", p1_elo)
                print("Player B Elo:", p2_elo)
                print(f"{i + 1} / {n_games} \n")

            # switch the player side
            player_2, player_1 = player_1, player_2
            turns = 1 - turns




    except KeyboardInterrupt:
        print('\n\rquit')
