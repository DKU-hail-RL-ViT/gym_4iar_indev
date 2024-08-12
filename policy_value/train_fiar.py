import argparse
import datetime
import random
import os

import numpy as np
import torch
import wandb

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

# from policy_value.policy_value_mcts_pure import RandomAction

# DQN, QRDQN, AC, QAC, QRAC, QRQAC, EQRDQN, EQRQAC
parser = argparse.ArgumentParser()

""" tuning parameter """
parser.add_argument("--n_playout", type=int, default=2)  # compare with 2, 10, 50, 100, 400
parser.add_argument("--quantiles", type=int, default=16)  # compare with 2, 16, 32, 64

""" RL model """
parser.add_argument("--rl_model", type=str, default="DQN")  # action value ver                  # Done
# parser.add_argument("--rl_model", type=str, default="QRDQN")  # action value ver
# parser.add_argument("--rl_model", type=str, default="AC")       # Actor critic state value ver    # Done
# parser.add_argument("--rl_model", type=str, default="QAC")    # Actor critic action value ver      # Done
# parser.add_argument("--rl_model", type=str, default="QRAC")   # Actor critic state value ver      # Done
# parser.add_argument("--rl_model", type=str, default="QRQAC")  # Actor critic action value ver
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
parser.add_argument("--batch_size", type=int, default=64)  # previous 64
parser.add_argument("--learn_rate", type=float, default=5e-4)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)
""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

"""DQN epsilon parameter """
parser.add_argument('--epsilon', type=float, default=0.1, help='Initial epsilon value for exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Decay rate for epsilon')
parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum value for epsilon after decay')

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
epsilon_decay = args.epsilon_decay
min_epsilon = args.min_epsilon


def get_equi_data(env, play_data):
    """augment the data set by rotating 180 degrees and flipping both horizontally and vertically
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_prob, winner in play_data:
        board_height, board_width = env.state_.shape[1], env.state_.shape[2]

        # Original state and MCTS probabilities
        extend_data.append((state, mcts_prob.flatten(), winner))

        # Rotate 180 degrees
        equi_state_180 = np.array([np.rot90(s, 2) for s in state])
        equi_mcts_prob_180 = np.rot90(mcts_prob.reshape(board_height, board_width), 2)
        extend_data.append((equi_state_180, equi_mcts_prob_180.flatten(), winner))

        # Flip horizontally
        equi_state_hor = np.array([np.fliplr(s) for s in state])
        equi_mcts_prob_hor = np.fliplr(mcts_prob.reshape(board_height, board_width))
        extend_data.append((equi_state_hor, equi_mcts_prob_hor.flatten(), winner))

        # Flip vertically
        equi_state_ver = np.array([np.flipud(s) for s in state])
        equi_mcts_prob_ver = np.flipud(mcts_prob.reshape(board_height, board_width))
        extend_data.append((equi_state_ver, equi_mcts_prob_ver.flatten(), winner))

    return extend_data


def collect_selfplay_data(mcts_player, game_iter, n_games=100):
    # self-play 100 games and save in data_buffer(queue)
    # in data_buffer store all steps of self-play so, it should be large enough
    data_buffer = deque(maxlen=36 * n_games * 4)  # board size * n_games * augmentation times
    win_cnt = defaultdict(int)

    for self_play_i in range(n_games):
        rewards, play_data = self_play(env, mcts_player, temp, game_iter, self_play_i)
        play_data = list(play_data)[:]
        # augment the data
        play_data = get_equi_data(env, play_data)
        data_buffer.extend(play_data)

        if rewards == -0.5:
            rewards = 0
        win_cnt[rewards] += 1

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("\n ---------- Self-Play win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    print("Win rate : ", round(win_ratio * 100, 3), "%")
    wandb.log({"Self-Play Win Ratio": round(win_ratio * 100, 3)})

    return data_buffer


def self_play(env, mcts_player, temp=1e-3, game_iter=0, self_play_i=0):
    obs, _ = env.reset()
    states, mcts_probs, current_player = [], [], []

    player_0 = 0
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    while True:
        move, move_probs = mcts_player.get_action(env, temp, return_prob=1)

        # store the data
        states.append(obs_post.copy())
        mcts_probs.append(move_probs)
        current_player.append(turn(env.state_))

        obs, reward, terminated, info = env.step(move)

        player_0 = turn(env.state_)
        player_1 = 1 - player_0
        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.self_play_winner()

        if end:
            if len(current_player) == 36 and winners == -1:  # draw
                print('self_play_draw')

            obs, _ = env.reset()  # reset env
            mcts_player.reset_player()  # reset MCTS root node

            print("game: {}, self_play:{}, episode_len:{}".format(
                game_iter + 1, self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if winners != -1:  # non draw
                if winners == -0.5:  # when win white player adjust to 0
                    winners = 0
                # if winner is current player, winner_z = 1
                winners_z[np.array(current_player) == 1 - winners] = 1.0
                winners_z[np.array(current_player) != 1 - winners] = -1.0

            return winners, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net, data_buffers=None):
    k, kl, loss, entropy = 0, 0, 0, 0
    lr_multiplier = lr_mul
    update_data_buffer = [data for buffer in data_buffers for data in buffer]

    """update the policy-value net"""
    mini_batch = random.sample(update_data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for k in range(epochs):
        loss, entropy = policy_value_net.train_step(state_batch,
                                                    mcts_probs_batch,
                                                    winner_batch,
                                                    learn_rate * lr_multiplier)

        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1))
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break

    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{}"
           ).format(kl, lr_multiplier, loss, entropy))
    return loss, entropy, lr_multiplier, policy_value_net


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    # leaf_mcts_player = MCTS_leaf(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    # random_action_player = RandomAction()
    win_cnt = defaultdict(int)

    for j in range(n_games):
        # reset for each game
        winner = start_play(env, training_mcts_player, opponent_mcts_player)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    wandb.log({"Evaluate Win Ratio": round(win_ratio * 100, 3)})
    return win_ratio, training_mcts_player


def start_play(env, player1, player2, move=None):
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
        move = player_in_turn.get_action(env, temp=1e-3, return_prob=0)  # self-play temp=1.0, eval temp=1e-3
        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            obs, _ = env.reset()
            return winner


if __name__ == '__main__':
    # wandb intialize
    # DQN, QRDQN, AC, QAC, QRAC, QRQAC, EQRDQN, EQRQAC
    if rl_model == "DQN" or rl_model == "AC" or rl_model == "QAC":
        wandb.init(mode="offline",
                   entity="hails",
                   project="gym_4iar",
                   name="FIAR-" + rl_model + "-MCTS" + str(n_playout) +
                        "-Date" + str(datetime.datetime.now()),
                   config=args.__dict__
                   )
    elif (rl_model == "QRDQN" or rl_model == "QRAC" or rl_model == "QRQAC"
          or rl_model == "EQRDQN" or rl_model == "EQRQAC"):
        wandb.init(mode="offline",
                   entity="hails",
                   project="gym_4iar",
                   name="FIAR-" + rl_model + "-MCTS" + str(n_playout) + "-Quantiles" + str(quantiles) +
                        "-Date" + str(datetime.datetime.now()),
                   config=args.__dict__
                   )
    else:
        assert False, "Model is not defined"

    env = Fiar()
    obs, _ = env.reset()

    if torch.cuda.is_available():           # Windows
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # Mac OS
        device = torch.device("mps")
    else:                                   # CPU
        device = torch.device("cpu")

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    # initialize Network
    if init_model:
        policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                          quantiles, model_file=init_model, rl_model=rl_model)
    else:
        policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                          quantiles, rl_model=rl_model)

    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                                               is_selfplay=1,  rl_model=rl_model)
    # curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
    #                               epsilon, epsilon_decay, min_epsilon, is_selfplay=1,  rl_model=rl_model)
    data_buffer_training_iters = deque(maxlen=20)
    best_old_model = None

    try:
        for i in range(training_iterations):
            """collect self-play data each iteration 100 games"""
            data_buffer_each = collect_selfplay_data(curr_mcts_player, i, self_play_sizes)  # 100 times
            data_buffer_training_iters.append(data_buffer_each)

            """Policy update with data buffer"""
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer_training_iters)
            wandb.log({"loss": loss, "entropy": entropy})

            if i == 0:
                """make mcts agent training, eval version"""
                policy_evaluate(env, curr_mcts_player, curr_mcts_player)

                if rl_model == "DQN" or rl_model == "AC" or rl_model == "QAC":
                    model_file = f"Training/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(model_file)
                    eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(eval_model_file)

                elif (rl_model == "QRDQN" or rl_model == "QRAC" or rl_model == "QRQAC"
                      or rl_model == "EQRDQN" or rl_model == "EQRQAC"):
                    model_file = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(model_file)
                    eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(eval_model_file)

                else:
                    assert False, "Model is not defined"

            else:
                if rl_model == "DQN" or rl_model == "AC" or rl_model == "QAC":
                    existing_files = [int(file.split('_')[-1].split('.')[0])
                                      for file in os.listdir(f"Training/{rl_model}_nmcts{n_playout}")
                                      if file.startswith('train_')]
                    old_i = max(existing_files)
                    best_old_model = f"Training/{rl_model}_nmcts{n_playout}/train_{old_i:03d}.pth"

                elif (rl_model == "QRDQN" or rl_model == "QRAC" or rl_model == "QRQAC"
                      or rl_model == "EQRDQN" or rl_model == "EQRQAC"):
                    existing_files = [int(file.split('_')[-1].split('.')[0])
                                      for file in
                                      os.listdir(f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}")
                                      if file.startswith('train_')]
                    old_i = max(existing_files)
                    best_old_model = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{old_i:03d}.pth"
                else:
                    assert False, "Model is not defined"

                policy_value_net_old = PolicyValueNet(env.state_.shape[1], env.state_.shape[2], quantiles,
                                                      best_old_model, rl_model=rl_model)

                """The most recent model with the highest win rate among the trained models"""
                old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout,
                                             is_selfplay=0, rl_model=rl_model)

                """Training model"""
                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout,
                                              is_selfplay=0, rl_model=rl_model)

                win_ratio, curr_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player)

                if (i + 1) % 10 == 0:  # save model 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 (1+10 : total 11)
                    if rl_model == "DQN" or rl_model == "AC" or rl_model == "QAC":
                        eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                        policy_value_net.save_model(eval_model_file)

                    elif (rl_model == "QRDQN" or rl_model == "QRAC" or rl_model == "QRQAC"
                          or rl_model == "EQRDQN" or rl_model == "EQRQAC"):
                        eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                        policy_value_net.save_model(eval_model_file)
                    else:
                        assert False, "don't have model"

                print("Win rate : ", round(win_ratio * 100, 3), "%")
                wandb.log({"Evaluation Win Rate": round(win_ratio * 100, 3)})

                if win_ratio > 0.5:
                    old_mcts_player = curr_mcts_player
                    if rl_model == "DQN" or rl_model == "AC" or rl_model == "QAC":
                        model_file = f"Training/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    elif (rl_model == "QRDQN" or rl_model == "QRAC" or rl_model == "QRQAC"
                          or rl_model == "EQRDQN" or rl_model == "EQRQAC"):
                        model_file = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                    else:
                        assert False, "don't have model"
                    policy_value_net.save_model(model_file)
                    print(" ---------- New best policy!!! ---------- ")

                else:
                    # if worse it just reject and does not go back to the old policy
                    print(" ---------- Low win-rate ---------- ")

    except KeyboardInterrupt:
        print('\n\rquit')
