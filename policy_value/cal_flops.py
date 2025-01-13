
import argparse
import random

import numpy as np
import torch

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer
from policy_value.efficient_mcts import EMCTSPlayer
from policy_value.file_utils import *

from thop import profile
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()

""" tuning parameter """
parser.add_argument("--n_playout", type=int, default=100)  # compare with 2, 10, 50, 100, 400
parser.add_argument("--quantiles", type=int, default=3)  # compare with 3, 9, 27, 81
parser.add_argument('--epsilon', type=float, default=0.7)  # compare with 0.1, 0.4, 0.7

"""Efficient Search Hyperparameter"""
# EQRDQN (2, 5832), (10, 29160), (50, 145800), (100, 291600),(400, 1166400)
# EQRQAC (2, 5832), (10, 29160), (50, 145800), (100, 291600),(400, 1166400)

parser.add_argument('--effi_n_playout', type=int, default=2)
parser.add_argument('--search_resource', type=int, default=5832)

""" RL model """
# parser.add_argument("--rl_model", type=str, default="DQN")  # action value ver
# parser.add_argument("--rl_model", type=str, default="QRDQN")  # action value ver
parser.add_argument("--rl_model", type=str, default="AC")       # Actor critic state value ver
# parser.add_argument("--rl_model", type=str, default="QAC")  # Actor critic action value ver
# parser.add_argument("--rl_model", type=str, default="QRAC")   # Actor critic state value ver
# # parser.add_argument("--rl_model", type=str, default="QRQAC")  # Actor critic action value ver
# parser.add_argument("--rl_model", type=str, default="EQRDQN") # Efficient search + action value ver
# parser.add_argument("--rl_model", type=str, default="EQRQAC")  # Efficient search + Actor critic action value ver

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=100)
parser.add_argument("--temp", type=float, default=1.0)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learn_rate", type=float, default=1e-3)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)

""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

args = parser.parse_args()

# make all args to variables
n_playout = args.n_playout
buffer_size = args.buffer_size
c_puct = args.c_puct
epochs = args.epochs
self_play_sizes = args.self_play_sizes
training_iterations = args.training_iterations
temp = args.temp
batch_size = args.batch_size
learn_rate = args.learn_rate
lr_mul = args.lr_mul
lr_multiplier = args.lr_multiplier
kl_targ = args.kl_targ
win_ratio = args.win_ratio
init_model = args.init_model
rl_model = args.rl_model
quantiles = args.quantiles
epsilon = args.epsilon
search_resource = args.search_resource
effi_n_playout = args.effi_n_playout


def measure_mcts_flops(env, mcts_player, policy_value_net_wrapper,
                       single_forward_flops, game_iter=0, temp=1.0, return_prob=1):
    """
    mcts_player.get_action(env, ...) 한 번 호출하는 동안의 총 FLOPs 추정치 계산
    """
    policy_value_net_wrapper.reset_count()
    move, move_probs = mcts_player.get_action(env, game_iter, temp, return_prob=return_prob)

    # 3) 최종 호출 횟수 * 단일 forward FLOPs
    total_call_count = policy_value_net_wrapper.count
    total_flops = total_call_count * single_forward_flops

    return move, move_probs, total_flops


def get_single_forward_flops(model, input_shape=(1, 4, 6, 7)):

    dummy_input = torch.randn(input_shape)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    return flops, params



if __name__ == '__main__':

    env = Fiar()
    obs, _ = env.reset()

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    init_model = f"Eval/{rl_model}_nmcts{n_playout}/train_100.pth"
    policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                      quantiles, model_file=init_model, rl_model=rl_model)

    input_size = (1, 5, 9, 4)
    single_forward_flops, params = get_single_forward_flops(policy_value_net.policy_value_net, input_size)
    print(single_forward_flops)

    if rl_model in ["EQRDQN", "EQRQAC"]:
        curr_mcts_player = EMCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                       epsilon, search_resource, is_selfplay=1, rl_model=rl_model)
    elif rl_model in ["DQN", "QRDQN", "QRAC", "QRQAC"]:
        curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, quantiles,
                                      epsilon, is_selfplay=1, rl_model=rl_model)
    else:
        curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                      epsilon, is_selfplay=1, rl_model=rl_model)



    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")