import os
import numpy as np

from fiar_env import Fiar, action2d_ize
from matplotlib import pyplot as plt
from PIL import Image
from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet


def policy_evaluate(env, player_1, player_2, n_games=2):
    """Evaluate the trained policy by playing games against the pure MCTS player"""
    winner = 0

    for i in range(n_games):
        winner = start_play(env, player_1, player_2, is_shown=1)

        if winner == 1:
            print("Black wins! \n\n")
        elif winner == -0.5:
            print("White wins! \n\n")
        else:
            print("Draw! \n\n")

        # switch the player side
        player_1, player_2 = player_2, player_1


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
        draw_board(env.state_, player1, player2, move)  # print the game board
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            obs, _ = env.reset()
            return winner


def draw_board(state, player1, player2, last_move=None):
    width, height = state.shape[1], state.shape[2]  # height = 9, width = 4
    dpi = 100
    fig, ax = plt.subplots()


    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(width), minor=True)
    ax.set_yticks(np.arange(height), minor=False)
    ax.grid(which='major', color='black', linestyle='-', linewidth=1)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='both', size=1, labelbottom=True, labelleft=True)  # remove numbers on axis
    ax.invert_yaxis()

    if last_move is not None:
        last_move = action2d_ize(last_move)
        print(last_move)
        # ax.plot(last_move[0], height - last_move[1] - 1, 'o', color='black', markersize=10)

    for j in range(height):
        for i in range(width):
            if state[3, i, j] == 1 and state[0, i, j] == 1:
                ax.text(j, height - i - 1, 'X', ha='center', va='center', fontsize=20)
            elif state[3, i, j] == 1 and state[1, i, j] == 1:
                ax.text(j, height - i - 1, 'O', ha='center', va='center', fontsize=20)
            elif state[3, i, j] == 0:
                ax.text(j, height - i - 1, ' ', ha='center', va='center', fontsize=20)

    print(player1, player2)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height())
    plt.close(fig)
    return image


def save_gif(images, filename):
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)


def create_game_gif(game_steps, player1, player2):
    images = []
    for state, last_move in game_steps:
        img_array = draw_board(state, player1, player2, last_move)
        img = Image.fromarray(img_array)
        images.append(img)
    save_gif(images, 'game.gif')


if __name__ == '__main__':
    env = Fiar()
    c_puct = 5

    # player 1 info
    p1_rl_model = "AC"
    p1_n_playout = 10
    p1_quantiles = 32
    p1_file_num = 1

    # player 2 info
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

    # fig, ax = plt.subplots()
    # 0 ~ 3 0 ~ 8

    policy_evaluate(env, player1, player2)

