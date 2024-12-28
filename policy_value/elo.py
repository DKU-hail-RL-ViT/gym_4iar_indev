from fiar_env import Fiar
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer
from policy_value.efficient_mcts import EMCTSPlayer

import argparse
import csv
import os
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--init_elo', type=int, default=1500)  # initial Elo rating
parser.add_argument('--k_factor', type=int, default=20)  # sensitivity of the rating adjustment
parser.add_argument('--c_puct', type=int, default=5)
args = parser.parse_args()

init_elo = args.init_elo
k_factor = args.k_factor
c_puct = args.c_puct


def create_player(model, playout=None, search_resource=None, quantile=None, epsilon=None):
    """
    MCTSPlayer ??? ???? ??.

    Args:
        model (str): RL model ("AC", "QAC", "QRAC", "QRQAC", "DQN", "QRDQN" ? ??)
        playout (int): number of playouts
        quantile (int, optional): num of Quantile
        epsilon (float, optional): Epsilon value

    Returns:
        player: ??? MCTSPlayer object
    """
    model_file = f"Eval/{model}"

    if model in ["EQRDQN", "EQRQAC"]:
        model_file += f"_resource{search_resource}"
    if playout is not None:
        model_file += f"_nmcts{playout}"
    if quantile is not None:
        model_file += f"_quantiles{quantile}"
    if epsilon is not None:
        model_file += f"_eps{epsilon}"

    model_file += "/train_100.pth"

    if not os.path.exists(model_file):
        return None

    policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                      quantile if model in ["QRAC", "QRQAC", "QRDQN"]
                                      else (81 if model in ["EQRDQN", "EQRQAC"] else None),
                                      model_file=model_file, rl_model=model
                                      )

    if model in ["DQN", "QRDQN", "QRAC", "QRQAC"]:
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout, quantile, epsilon,
                            is_selfplay=0, rl_model=model
                            )

    elif model in ["AC", "QAC"]:
        player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout, epsilon,
                            is_selfplay=0, rl_model=model
                            )

    else:  # EQRQAC, EQRDQN
        playout = 40000
        player = EMCTSPlayer(policy_value_net.policy_value_fn, c_puct, playout,
                             epsilon if model in ["EQRDQN"] else None, search_resource,
                             is_selfplay=0, rl_model=model
                             )

    player.name = model_file

    return player


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

    if winner == 1:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo)
    elif winner == -1:
        player2.elo, player1.elo = update_elo(player2.elo, player1.elo)
    elif winner == 0:
        player1.elo, player2.elo = update_elo(player1.elo, player2.elo, draw=True)
    else:
        assert False


if __name__ == '__main__':

    wandb.init(entity="hails",
               project="gym_4iar_elo",
               name="elo_test",
               config=args.__dict__
               )

    env = Fiar()
    obs, _ = env.reset()

    models = ["AC", "QRAC", "QAC", "QRQAC", "DQN", "QRDQN", "EQRDQN", "EQRQAC"]
    num_playout = [2, 10, 50, 100, 400]
    num_quantile = [3, 9, 27, 81]
    num_epsilon = [0.1, 0.4, 0.7]

    search_resource_qrdqn = [5913, 26325, 120366, 246969, 951426, 5913, 26325, 137376, 271512, 1066365,
                             5913, 26325, 139239, 278073, 1065717]
    index_slices = [5, 10, 15]

    search_resource_qrqac = [5913, 29231, 144828, 286578, 1137078]

    player_list = []
    model_counts = {model: 0 for model in models}

    for model in models:
        if model in ["AC", "QAC"]:
            for playout in num_playout:
                player = create_player(model=model, playout=playout)
                player_list.append(player)
                model_counts[model] += 1

        elif model in ["QRAC", "QRQAC"]:
            for playout in num_playout:
                for quantile in num_quantile:
                    player = create_player(model=model, playout=playout, quantile=quantile)
                    player_list.append(player)
                    model_counts[model] += 1

        elif model in ["DQN"]:
            for playout in num_playout:
                for epsilon in num_epsilon:
                    player = create_player(model=model, playout=playout, epsilon=epsilon)
                    player_list.append(player)
                    model_counts[model] += 1

        elif model in ["QRDQN"]:
            for playout in num_playout:
                for quantile in num_quantile:
                    for epsilon in num_epsilon:
                        player = create_player(model=model, playout=playout, quantile=quantile, epsilon=epsilon)
                        player_list.append(player)
                        model_counts[model] += 1

        elif model == "EQRDQN":
            for i, epsilon in enumerate(num_epsilon):
                start_idx = index_slices[i - 1] if i > 0 else 0
                end_idx = index_slices[i]

                for resource in search_resource_qrdqn[start_idx:end_idx]:
                    player = create_player(model=model, search_resource=resource, epsilon=epsilon)
                    player_list.append(player)
                    model_counts[model] += 1

        elif model == "EQRQAC":
            for resource in search_resource_qrqac:
                player = create_player(model=model, search_resource=resource)
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
    os.makedirs('./gamefile', exist_ok=True)
    with open('./gamefile/player_elo_result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Player Name", "Elo Rating"])
        for player in player_list:
            writer.writerow([player.name, player.elo])