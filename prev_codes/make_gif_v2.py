import os
import torch
import numpy as np
from fiar_env import Fiar, action2d_ize
from matplotlib import pyplot as plt
from matplotlib import animation
from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet


def policy_evaluate(env, player_1, player_2, n_games=2):
    """Evaluate the trained policy by playing games against the pure MCTS player"""
    winner = 0
    frames = []  # 이미지 프레임 저장을 위한 리스트

    for i in range(n_games):
        winner, game_frames = start_play(env, player_1, player_2)
        frames.extend(game_frames)  # 게임 프레임 추가

        if winner == 1: # [Todo] 에러 수정
            print("Black wins! \n\n")
        elif winner == -0.5:
            print("White wins! \n\n")
        else:
            print("Draw! \n\n")

        # switch the player side
        player_1, player_2 = player_2, player_1

    create_gif(frames)  # GIF 생성


def create_gif(frames):
    """Create a GIF from a list of frames"""
    fig, ax = plt.subplots()
    patch = plt.imshow(frames[0], cmap='viridis', animated=True)

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=10)
    anim.save('game.gif', writer='pillow', fps=30)
    plt.close()


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
    frames = []  # 게임 진행 중의 모든 프레임을 저장

    while True:
        frame = graphic(env, move)  # 게임 보드 이미지 저장
        frames.append(frame)  # 프레임 추가

        move = players[current_player].get_action(env, temp=0.1, return_prob=0)
        obs, reward, terminated, info = env.step(move)

        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
        else:
            obs, _ = env.reset()
            return winner, frames


def graphic(env, move=None, file_prefix='game_board'):
    """Draw the board and show game info as an image"""
    width = env.state_.shape[1]
    height = env.state_.shape[2]

    fig, ax = plt.subplots(figsize=(9,4))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(False)
    ax.xaxis.tick_top() # x축 label 위쪽에 위치
    plt.gca().invert_yaxis()  # y축 순서 뒤집기

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
    print("wtf")

    save_num = 0
    folder_path = './vid/'
    # os.makedirs(os.path.dirname(model_file), exist_ok=True)
    file_path = os.path.join(folder_path, f'saved_png/{file_prefix}_{save_num}.png')

    os.path.join('../policy_value')

    # 만약 파일이 이미 존재하면, 번호를 증가시켜 가면서 새로운 파일명을 찾는다
    while os.path.exists(file_path):
        save_num += 1
        file_path = os.path.join(folder_path, f'saved_png/{file_prefix}_{save_num}.png')

    plt.savefig(file_path)
    print(f"Saved: {file_path}")
    plt.close()


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


    policy_evaluate(env, player1, player2)
