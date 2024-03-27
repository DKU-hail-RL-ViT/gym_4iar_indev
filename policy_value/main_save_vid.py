import os

from fiar_env import Fiar, action2d_ize
from matplotlib import pyplot as plt

from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet


def policy_evaluate(env, player_1, player_2, n_games=2):
    """Evaluate the trained policy by playing games against the pure MCTS player"""
    winner = 0

    for i in range(n_games):
        winner = start_play(env, player_1, player_2)

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
        graphic(env, player1, player2, current_player, move)    # print the game board
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            obs, _ = env.reset()
            return winner


def graphic(env, player1, player2, curr_player, move=None):
    """Draw the board and show game info"""
    width = env.state_.shape[1]
    height = env.state_.shape[2]

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
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


if __name__ == '__main__':
    env = Fiar()
    rl_model = 'AC'
    n_playout = 2
    c_puct = 5

    black = f"RL_{rl_model}_nmcts{n_playout}/train_088.pth"  # player 1
    white = f"RL_{rl_model}_nmcts{n_playout}/train_100.pth"  # player 2

    save_dir = './vid/'
    # os.makedirs(os.path.join(save_dir, black.split('.')[0]), exist_ok=True)  # test

    policy_value_net_b = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                                        black, rl_model=rl_model)
    policy_value_net_w = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                                        white, rl_model=rl_model)

    player1 = MCTSPlayer(policy_value_net_b.policy_value_fn, c_puct, n_playout, is_selfplay=0)
    player2 = MCTSPlayer(policy_value_net_w.policy_value_fn, c_puct, n_playout, is_selfplay=0)

    # fig, ax = plt.subplots()
    # 0 ~ 3 0 ~ 8

    policy_evaluate(env, player1, player2)

# for ep in range(1):
# initialize env
# ax.set_xlim(0, 4)
# ax.set_ylim(0, 9)
# ax.plot(0+0.5, 0+0.5, 'o', color='black', markersize=20)
# https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea
# https://stackoverflow.com/questions/25140952/matplotlib-save-animation-in-gif-error
# https://pinkwink.kr/860
