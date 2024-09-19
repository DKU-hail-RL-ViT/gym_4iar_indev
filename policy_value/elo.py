from fiar_env import Fiar
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--init_elo', type=int, default=1500)  # initial Elo rating
parser.add_argument('--k_factor', type=int, default=20)  # sensitivity of the rating adjustment
parser.add_argument('--c_puct', type=int, default=5)
args = parser.parse_args()

init_elo = args.init_elo
k_factor = args.k_factor
c_puct = args.c_puct


def create_player(model, playout, quantile=None, epsilon=None):
    """
    MCTSPlayer 객체를 생성하는 함수.

    Args:
        model (str): RL model ("AC", "QAC", "QRAC", "QRQAC", "DQN", "QRDQN" 중 하나)
        playout (int): number of playouts
        quantile (int, optional): num of Quantile
        epsilon (float, optional): Epsilon value

    Returns:
        player: 생성된 MCTSPlayer object
    """
    model_file = f"Eval/{model}_nmcts{playout}"
    if quantile is not None:
        model_file += f"_quantiles{quantile}"
    if epsilon is not None:
        model_file += f"_eps{epsilon}"
    model_file += "/train_100.pth"

    if not os.path.exists(model_file):
        return None

    policy_value_net = PolicyValueNet(
        env.state().shape[1],
        env.state().shape[2],
        quantile if model in ["QRAC", "QRQAC", "QRDQN"] else None,
        model_file=model_file,
        rl_model=model
    )

    player = MCTSPlayer(
        policy_value_net.policy_value_fn,
        c_puct,
        playout,
        epsilon if model in ["DQN", "QRDQN"] else None,
        is_selfplay=0,
        rl_model=model
    )

    player.name = f"{model}_nmcts{playout}"
    if quantile is not None:
        player.name += f"_quantiles{quantile}"
    if epsilon is not None:
        player.name += f"_eps{epsilon}"

    return player


def wins(winner):
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
    player_in_turn = players[current_player]

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, temp=1e-3, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

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

    models = ["AC", "QRAC", "QAC", "QRQAC", "DQN", "QRDQN"]
    num_playout = [2, 10, 50, 100, 400]
    num_quantile = [3, 9, 27, 81]
    num_epsilon = [0.1, 0.4, 0.7]
    player_list = []
    model_counts = {model: 0 for model in models}

    for model in models:
        if model in ["AC", "QAC"]:
            for playout in num_playout:
                player = create_player(model, playout)
                if player:
                    player_list.append(player)
                    model_counts[model] += 1

        elif model in ["QRAC", "QRQAC"]:
            for playout in num_playout:
                for quantile in num_quantile:
                    player = create_player(model, playout, quantile=quantile)
                    if player:
                        player_list.append(player)
                        model_counts[model] += 1

        elif model in ["DQN"]:
            for playout in num_playout:
                for epsilon in num_epsilon:
                    player = create_player(model, playout, epsilon=epsilon)
                    if player:
                        player_list.append(player)
                        model_counts[model] += 1

        elif model in ["QRDQN"]:
            for playout in num_playout:
                for quantile in num_quantile:
                    for epsilon in num_epsilon:
                        player = create_player(model, playout, quantile=quantile, epsilon=epsilon)
                        if player:
                            player_list.append(player)
                            model_counts[model] += 1


        else:
            raise RuntimeError("Unexpected model type.")

    for model, count in model_counts.items():
        print(f"Number of {model} models: {count}")

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