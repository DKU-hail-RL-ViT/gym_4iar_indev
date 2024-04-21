import os
import torch

from fiar_env import Fiar, action2d_ize
from matplotlib import pyplot as plt

from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet


def wins(winner, player_1, player_2, start=0):
    if start == 0:
        if winner == 1:
            print("Black wins! \n")
        elif winner == -0.5:
            print("White wins! \n")
        else:
            print("Draw! \n")
    else:
        if winner == 1:
            print("Black wins! \n")
        elif winner == -0.5:
            print("White wins! \n")
        else:
            print("Draw! \n")
    print("Black : ", player_1)
    print("White : ", player_2,"\n\n")


def policy_evaluate(env, player_1, player_2, start=0, winner=0):

    winner = start_play(env, player_1, player_2)
    wins(winner, player_1, player_2, start)

    start = 1 - start
    winner = start_play(env, player_2, player_1)
    wins(winner, player_1, player_2, start)


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
        graphic(env, current_player, move)  # print the game board
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            obs, _ = env.reset()
            return winner


def graphic(env, curr_player, move=None):
    """Draw the board and show game info"""
    width = env.state_.shape[1]
    height = env.state_.shape[2]

    if p1_rl_model == "AC":
        p1 = f"{p1_rl_model}_nmcts{p1_n_playout}/train_{p1_file_num:03d}"
    else:
        p1 = f"{p1_rl_model}_nmcts{p1_n_playout}_quantiles{p1_quantiles}/train_{p1_file_num:03d}"
    if p2_rl_model == "AC":
        p2 = f"{p2_rl_model}_nmcts{p2_n_playout}/train_{p2_file_num:03d}"
    else:
        p2 = f"{p2_rl_model}_nmcts{p2_n_playout}_quantiles{p2_quantiles}/train_{p2_file_num:03d}"

    print("Player", p1, "with O".rjust(3))
    print("Player", p2, "with X".rjust(3))
    print()
    for x in range(width):
        print("{0:8}".format(x), end='')
    print('\r\n')

    if move is not None:
        action2d = action2d_ize(move)

    for i in range(height - 1, -1, -1):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            if move is not None and action2d == [j, i]:
                if curr_player == 0:
                    print('O'.center(8), end='')
                else:
                    print('X'.center(8), end='')
            else:
                if env.state_[0][j][i] == 1.0:
                    print('O'.center(8), end='')
                elif env.state_[1][j][i] == 1.0:
                    print('X'.center(8), end='')
                else:
                    print('_'.center(8), end='')
        print('\r\n\r\n')
    print("hello world")

if __name__ == '__main__':
    env = Fiar()
    c_puct = 5

    # player 1
    p1_rl_model = "AC"
    p1_n_playout = 10
    p1_quantiles = 32
    p1_file_num = 1

    # player 2
    p2_rl_model = "QRAC"
    p2_n_playout = 400
    p2_quantiles = 32
    p2_file_num = 1

    if p1_rl_model == "AC":
        p1 = f"Eval/{p1_rl_model}_nmcts{p1_n_playout}/train_{p1_file_num:03d}.pth"
    else:
        p1 = f"Eval/{p1_rl_model}_nmcts{p1_n_playout}_quantiles{p1_quantiles}/train_{p1_file_num:03d}.pth"
    if p2_rl_model == "AC":
        p2 = f"Eval/{p2_rl_model}_nmcts{p2_n_playout}/train_{p2_file_num:03d}.pth"
    else:
        p2 = f"Eval/{p2_rl_model}_nmcts{p2_n_playout}_quantiles{p2_quantiles}/train_{p2_file_num:03d}.pth"

    # save_dir = './vid/'
    # os.makedirs(os.path.join(save_dir, black.split('.')[0]), exist_ok=True)  # test

    p1_net = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                            p1_quantiles, model_file=p1, rl_model=p1_rl_model)
    p2_net = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                            p2_quantiles, model_file=p2, rl_model=p2_rl_model)

    player1 = MCTSPlayer(p1_net.policy_value_fn, c_puct, p1_n_playout, is_selfplay=0)
    player2 = MCTSPlayer(p2_net.policy_value_fn, c_puct, p2_n_playout, is_selfplay=0)

    policy_evaluate(env, player1, player2)


# fig, ax = plt.subplots()
    # 0 ~ 3 0 ~ 8

# for ep in range(1):
# initialize env
# ax.set_xlim(0, 4)
# ax.set_ylim(0, 9)
# ax.plot(0+0.5, 0+0.5, 'o', color='black', markersize=20)
# https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea
# https://stackoverflow.com/questions/25140952/matplotlib-save-animation-in-gif-error
# https://pinkwink.kr/860
