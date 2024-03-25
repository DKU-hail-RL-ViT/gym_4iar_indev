#
import torch
import os
from collections import defaultdict

from fiar_env import Fiar
from matplotlib import pyplot as plt

from policy_value.mcts import MCTSPlayer
from policy_value.policy_value_network import PolicyValueNet




def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):
	"""Evaluate the trained policy by playing games against the pure MCTS player"""
	current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
									 c_puct=self.c_puct,
									 n_playout=self.n_playout)
	pure_mcts_player = MCTS_Pure(c_puct=5,
								 n_playout=self.pure_mcts_playout_num)
	win_cnt = defaultdict(int)
	for i in range(n_games):
		winner = self.game.start_play(current_mcts_player,
									  pure_mcts_player,
									  start_player=i % 2,
									  is_shown=1)
		win_cnt[winner] += 1
	win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
	print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
		self.pure_mcts_playout_num,
		win_cnt[1], win_cnt[2], win_cnt[-1]))
	return win_ratio

def graphic(board, player1, player2):
	"""Draw the board and show game info"""
	width = board.width
	height = board.height

	print("Player", player1, "with X".rjust(3))
	print("Player", player2, "with O".rjust(3))
	print()
	for x in range(width):
		print("{0:8}".format(x), end='')
	print('\r\n')
	for i in range(height - 1, -1, -1):
		print("{0:4d}".format(i), end='')
		for j in range(width):
			loc = i * width + j
			p = board.states.get(loc, -1)
			if p == player1:
				print('X'.center(8), end='')
			elif p == player2:
				print('O'.center(8), end='')
			else:
				print('_'.center(8), end='')
		print('\r\n\r\n')


if __name__ == '__main__':

	env = Fiar()
	rl_model = 'AC'
	n_playout = 100

	black = f"RL_{rl_model}_nmcts{n_playout}/001.pth"  # player 1
	white = f"RL_{rl_model}_nmcts{n_playout}/012.pth"  # player 2

	save_dir = './vid/'
	os.makedirs(os.path.join(save_dir, file_name.split('.')[0]), exist_ok=True)

	# load file_name
	model_b = torch.load(black, map_location='cpu')
	model_w = torch.load(white, map_location='cpu')

	policy_value_net_b= PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
										  model_b, rl_model=rl_model)
	policy_value_net_w = PolicyValueNet(env.state_.shape[1], env.state_.shape[2],
										  model_w, rl_model=rl_model)

	curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)
	old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)

	fig, ax = plt.subplots()
	# 0 ~ 3 0 ~8


	policy_evaluate(env, model_b, model_w)

# for ep in range(1):
	# initialize env
	# ax.set_xlim(0, 4)
	# ax.set_ylim(0, 9)
	# ax.plot(0+0.5, 0+0.5, 'o', color='black', markersize=20)
	# https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea
	# https://stackoverflow.com/questions/25140952/matplotlib-save-animation-in-gif-error
	# https://pinkwink.kr/860


