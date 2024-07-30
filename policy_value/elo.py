from fiar_env import Fiar
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer
from itertools import product

import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--init_elo', type=int, default=1500)  # initial Elo rating
parser.add_argument('--k_factor', type=int, default=20)  # sensitivity of the rating adjustment
parser.add_argument('--c_puct', type=int, default=5)
args = parser.parse_args()

init_elo = args.init_elo
k_factor = args.k_factor
c_puct = args.c_puct


def wins(winner, result=0):
    """ black standard win = 1, draw = 0.5 ,lose = 0 """
    if winner == 1:  # p1 = black & p1 wins
        result = 1
    elif winner == -0.5:  # p1 = black & p2 wins
        result = 0
    elif winner == -1:  # p1 = black & draw
        result = 0.5
    else:
        assert False  # should not reach here
    return result


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
        move = player_in_turn.get_action(env, temp=1e-3, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            obs, _ = env.reset()
            return winner


class Player:
    def __init__(self, name, elo=None):
        self.name = name
        self.elo = elo if elo is not None else init_elo  # Set ELO to 1500 if not specified


def update_elo(winner_elo, loser_elo, draw=False, k=k_factor):
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner
    score_winner = 0.5 if draw else 1
    score_loser = 0.5 if draw else 0

    new_winner_elo = round(winner_elo + k * (score_winner - expected_winner), 1)
    new_loser_elo = round(loser_elo + k * (score_loser - expected_loser), 1)
    return new_winner_elo, new_loser_elo


# Simulation of a game between two players
def simulate_game(player1, player2):
    # This is a stub function that randomly determines the outcome
    winner = start_play(env, player1, player2)
    result = wins(winner)
    if result == 1:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo)
    elif result == 0:
        player2.elo, player1.elo = update_elo(player2.elo, player1.elo)
    elif result == 0.5:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo, draw=True)
    else:
        assert False


if __name__ == '__main__':

    env = Fiar()
    obs, _ = env.reset()

    models = ["AC", "QRAC"]
    playouts = [2, 10, 50, 100, 400]
    file_nums = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    quantiles = [2, 16, 32, 64]
    player_list = []

    for model, playout, file_num in product(models, playouts, file_nums):
        if model == "AC":
            # AC 모델의 경우 quantile 없이 파일 경로 생성
            model_file = f"Eval/{model}_nmcts{playout}/train_{file_num:03d}.pth"
            policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                              model_file=model_file, rl_model="AC")
            player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout, is_selfplay=0)
            player.name = f"{model}_nmcts{playout}_train_{file_num:03d}"
            player_list.append(player)

        elif model == "QRAC":

            for quantile in quantiles:
                model_file = f"Eval/{model}_nmcts{playout}_quantiles{quantile}/train_{file_num:03d}.pth"
                policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2], quantile,
                                                  model_file=model_file, rl_model="QRAC")
                player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout, is_selfplay=0)
                player.name = f"{model}_nmcts{playout}_quantiles{quantile}_train_{file_num:03d}.pth"
                player_list.append(player)
        else:
            RuntimeError("wtf")

    # List to hold game results
    game_results = []

    # Simulate games between all pairs of players
    for i, player1 in enumerate(player_list):
        for player2 in player_list[i + 1:]:
            simulate_game(player1, player2)
            print(f"{player1.name} ({player1.elo}) vs {player2.name} ({player2.elo})")

    player_list.reverse()
    # or using slicing
    # player_list = player_list[::-1]

    for i, player1 in enumerate(player_list):
        for player2 in player_list[i + 1:]:
            simulate_game(player1, player2)
            print(f"{player1.name} ({player1.elo}) vs {player2.name} ({player2.elo})")

    player_list.reverse()

    # Write player_list information to CSV
    with open('./gamefile/player_elo_result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Player Name", "Elo Rating"])
        for player in player_list:
            writer.writerow([player.name, player.elo])
