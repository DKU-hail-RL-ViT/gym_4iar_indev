from __future__ import print_function
from fiar_env import Fiar, action2d_ize
from policy_value_network import PolicyValueNet
from mcts import MCTSPlayer

import numpy as np


def location_to_move(location, board_width=9, board_height=4):
    if len(location) != 2:
        return -1
    h = location[0]
    w = location[1]

    move = board_height * w + h
    if move not in range(board_width * board_height):
        return -1
    return move


def start_play(env, player1, player2, human_index=None):
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
    is_human_index = (current_player == human_index)

    while True:
        # synchronize the MCTS tree with the current state of the game
        if is_human_index:
            move = player_in_turn.get_action(env)
        else:  # 모델이 플레이어라면, 매개변수를 조정하여 get_action 호출
            move = player_in_turn.get_action(env, temp=0.1, return_prob=0)

        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            is_human_index = (current_player == human_index)
            player_in_turn = players[current_player]
            if current_player == 0:
                player_in_turn.oppo_node_update(move)  # [ToDo] 여기 수정 해야 함
        else:
            print(env)
            obs, _ = env.reset()
            if winner == 1:
                print("Black wins! \n\n")
            elif winner == -0.5:
                print("White wins! \n\n")
            else:
                print("Draw! \n\n")

            return winner


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        availables = [i for i in range(36) if not np.any(env.state()[3][i // 4][i % 4] == 1)]
        try:
            print(env)
            location = input("Your move (max35 = 3,8) : ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in availables:
            print("invalid move")
            move = self.get_action(env.state_)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(env, rl_model, n_playout, quantile, file_num, start_player=0):

    if rl_model == "AC":
        model_file = f"Eval/{rl_model}_nmcts{n_playout}/train_{file_num:03d}.pth"
    else:
        model_file = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantile}/train_{file_num:03d}.pth"

    try:
        best_policy = PolicyValueNet(env.state().shape[1], env.state().shape[2], quantile,
                                     model_file=model_file, rl_model=rl_model)

        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5,
                                 n_playout=n_playout)  # set larger n_playout for better performance

        # human player, input your move in the format: 2,3
        human = Human()

        if start_player == 0:  # set start_player=0 for human first
            player = 0
            start_play(env, human, mcts_player, player)
        else:
            player = 1
            start_play(env, mcts_player, human, player)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    env = Fiar()

    p1_rl_model = "QRAC"
    p1_n_playout = 400
    p1_quantiles = 32
    p1_file_num = 100
    start_player = 1  # start_player 0 : human(black), start_player 1 : human(white)

    run(env, p1_rl_model, p1_n_playout, p1_quantiles, p1_file_num, start_player)
