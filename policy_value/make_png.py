import os
import torch
import numpy as np
from fiar_env import Fiar, action2d_ize
from matplotlib import pyplot as plt
from matplotlib import animation
from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet


def check_game_type(env, player1, player2, human): 
    print("========== Select game type ==========")
    print("\t0 : AI vs AI")    
    print("\t1 : Human vs AI (Human first)")
    print("\t2 : AI vs Human (AI first)")
    print("======================================")
    game_type = int(input("Select game type (range int 0~2) : "))

    if game_type == 0:        
        policy_evaluate(env, player1, player2)    
    elif game_type == 1:        
        policy_evaluate(env, human, player1)    
    elif game_type == 2:        
        policy_evaluate(env, player1, human)    
    else:        
        print("Invalid input")        
        check_game_type(env, player1, player2, human)


def location_to_move(location, board_width=9, board_height=4):
    if len(location) != 2:
        return -1
    h = location[0]
    w = location[1]

    move = board_height * w + h
    if move not in range(board_width * board_height):
        return -1
    return move


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
        self.info = "human"

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env, temp=None, return_prob=None):
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


def policy_evaluate(env, player_1, player_2, n_games=1):
    """Evaluate the trained policy by playing games against the pure MCTS player"""
    winner = 0
    frames = []  # 이미지 프레임 저장을 위한 리스트

    for i in range(n_games):
        winner, game_frames = start_play(env, player_1, player_2)
        frames.extend(game_frames)  # 게임 프레임 추가

        if winner == 1:
            print("Black wins! \n\n")
        elif winner == -0.5:
            print("White wins! \n\n")
        else:
            print("Draw! \n\n")

    # create_gif(frames)  # GIF 생성


def create_gif(frames):
    """Create a GIF from a list of frames"""
    fig, ax = plt.subplots()
    patch = plt.imshow(frames[0], cmap='viridis', animated=True)

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=10)
    anim.save('game.gif', writer='pillow', fps=30)
    plt.close()


def start_play(env, player1=None, player2=None):
    """start a game between two players"""
    obs, _ = env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    move = None
    save_num = 0
    frames = []  # 게임 진행 중의 모든 프레임을 저장

    while True:
        graphic(env, save_num, move)
        save_num += 1
        # frame = graphic(env, save_num, move)  # 게임 보드 이미지 저장
        # frames.append(frame)  # 프레임 추가

        move = players[current_player].get_action(env, temp=0.1, return_prob=0)
        obs, reward, terminated, info = env.step(move)
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
        else:
            graphic(env, save_num, move)  # if end
            print(env)
            obs, _ = env.reset()
            return winner, frames


def graphic(env, save_num, move=None, file_prefix='game_board'):
    """Draw the board and show game info as an image"""
    width = env.state_.shape[1]
    height = env.state_.shape[2]

    if player1.info == "human":
        p1 = f"Human_player"
    elif p1_rl_model == "AC":
        p1 = f"{p1_rl_model}_nmcts{p1_n_playout}/train_{p1_file_num:03d}"
    elif p1_rl_model == "QRAC":
        p1 = f"{p1_rl_model}_nmcts{p1_n_playout}_quantiles{p1_quantiles}/train_{p1_file_num:03d}"
    else:
        assert False, "invaild"

    if player2.info == "human":
        p2 = f"Human_player"
    elif p2_rl_model == "AC":
        p2 = f"{p2_rl_model}_nmcts{p2_n_playout}/train_{p2_file_num:03d}"
    elif p2_rl_model == "QRAC":
        p2 = f"{p2_rl_model}_nmcts{p2_n_playout}_quantiles{p2_quantiles}/train_{p2_file_num:03d}"
    else:
        assert False, "invaild"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(False)
    ax.xaxis.tick_top()  # x축 label 위쪽에 위치
    plt.gca().invert_yaxis()  # y축 순서 뒤집기
    plt.rcParams.update({'font.family': 'DejaVu Sans'})  # [ToDO] font 바꾸고 싶은데 적용이 안됨..
    plt.xlabel(f"Black Player: O, Model : {p1} \nWhite Player: X, Model : {p2}")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(height):
        for j in range(width):
            symbol = ' '
            if env.state_[0][j][height - 1 - i] == 1.0:
                symbol = 'O'
            elif env.state_[1][j][height - 1 - i] == 1.0:
                symbol = 'X'
            ax.text(j, height - 1 - i, symbol, fontsize=12, ha='center', va='center')

    folder_path = './Png/'
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)

    file_path = os.path.join(folder_path, f'{file_prefix}_{save_num}.png')
    plt.savefig(file_path)
    print(f"Saved: {file_path}")
    plt.close()


if __name__ == '__main__':
    env = Fiar()
    c_puct = 5

    # player 1 info
    p1_rl_model = "QRAC"
    p1_n_playout = 400
    p1_quantiles = 32
    p1_file_num = 100

    # player 2 info
    p2_rl_model = "QRAC"
    p2_n_playout = 100
    p2_quantiles = 32
    p2_file_num = 50

    # human player, input your move in the format: 2,3
    human = Human()

    if p1_rl_model == "AC":
        p1 = f"Eval/{p1_rl_model}_nmcts{p1_n_playout}/train_{p1_file_num:03d}.pth"
    else:
        p1 = f"Eval/{p1_rl_model}_nmcts{p1_n_playout}_quantiles{p1_quantiles}/train_{p1_file_num:03d}.pth"
    if p2_rl_model == "AC":
        p2 = f"Eval/{p2_rl_model}_nmcts{p2_n_playout}/train_{p2_file_num:03d}.pth"
    else:
        p2 = f"Eval/{p2_rl_model}_nmcts{p2_n_playout}_quantiles{p2_quantiles}/train_{p2_file_num:03d}.pth"

    p1_net = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                            p1_quantiles, model_file=p1, rl_model=p1_rl_model)
    p2_net = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
                            p2_quantiles, model_file=p2, rl_model=p2_rl_model)

    player1 = MCTSPlayer(p1_net.policy_value_fn, c_puct, p1_n_playout, is_selfplay=0)
    player2 = MCTSPlayer(p2_net.policy_value_fn, c_puct, p2_n_playout, is_selfplay=0)

    check_game_type(env, player1, player2, human)
