import argparse
import datetime
import random
import os

import numpy as np
import torch
import wandb

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize, carculate_area
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

# from policy_value.policy_value_mcts_pure import RandomAction

parser = argparse.ArgumentParser()

""" tuning parameter """
parser.add_argument("--n_playout", type=int, default=400)  # compare with 2, 10, 50, 100, 400
parser.add_argument("--quantiles", type=int, default=32)  # compare with 2, 16, 32, 64

""" RL model """
# parser.add_argument("--rl_model", type=str, default="AC")
parser.add_argument("--rl_model", type=str, default="QRAC")
# parser.add_argument("--rl_model", type=str, default="EQRAC")

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=100)
parser.add_argument("--temp", type=float, default=0.1)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learn_rate", type=float, default=5e-4)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)
""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)
""" Quantile Regression parameter """

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


def policy_value_fn(board):  # board.shape = (9,4)
    # return uniform probabilities and 0 score for pure MCTS
    availables = [i for i in range(36) if not np.any(board[3][i // 4][i % 4] == 1)]
    action_probs = np.ones(len(availables)) / len(availables)
    return zip(availables, action_probs), 0


def get_equi_data(env, play_data):
    """augment the data set by flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_prob, winner in play_data:
        # flip horizontally
        equi_state = np.array([np.fliplr(s) for s in state])
        equi_mcts_prob = np.fliplr(mcts_prob.reshape(env.state_.shape[1], env.state_.shape[2]))
        extend_data.append((equi_state,
                            np.flipud(equi_mcts_prob).flatten(),
                            winner))
    return extend_data


def collect_selfplay_data(mcts_player, game_iter, n_games=100):
    # self-play 100 games and save in data_buffer(queue)
    # in data_buffer store all steps of self-play so, it should be large enough
    data_buffer = deque(maxlen=36 * n_games)
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

    player_0 = 0  # previous : turn(obs)
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    move = None
    move_probs = None

    while True:
        while True:
            move, move_probs = mcts_player.get_action(env, temp, return_prob=1)
            action2d = action2d_ize(move)
            if env.state_[3, action2d[0], action2d[1]] == 0.0:
                break

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
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            learn_rate * lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                     )
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
    # leaf_mcts_player = MCTS_leaf(policy_value_fn, c_puct=c_puct, n_playout=n_playout) # [TODO] leaf node MCTS
    # random_action_player = RandomAction()  # [TODO] random actions Agent
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
    return win_ratio, training_mcts_player


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
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

        else:
            wandb.log({"Reward": reward[1]})
            obs, _ = env.reset()
            return winner


if __name__ == '__main__':
    # wandb intialize
    if rl_model == "AC":
        wandb.init(mode="online",
                   entity="hails",
                   project="gym_4iar",
                   name="FIAR-" + rl_model + "-MCTS" + str(n_playout) +
                        "-Date" + str(datetime.datetime.now()),
                   config=args.__dict__
                   )

    elif rl_model == "QRAC":
        wandb.init(mode="online",
                   entity="hails",
                   project="gym_4iar",
                   name="FIAR-" + rl_model + "-MCTS" + str(n_playout) + "-Quantiles" + str(quantiles) +
                        "-Date" + str(datetime.datetime.now()),
                   config=args.__dict__
                   )
    elif rl_model == "EQRAC":
        assert False, "Not implemented yet"
    else:
        assert False, "don't have model"

    env = Fiar()
    obs, _ = env.reset()

    if torch.cuda.is_available():  # Windows
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Mac OS
        device = torch.device("mps")
    else:  # CPU
        device = torch.device("cpu")

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    if init_model:
        # start training from an initial policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2], quantiles,
                                          model_file=init_model, rl_model=rl_model)
    else:
        # start training from a new policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1], env.state().shape[2],
                                          quantiles, rl_model=rl_model)

    # policy_value_net_old = copy.deepcopy(policy_value_net)
    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                  c_puct,
                                  n_playout,
                                  is_selfplay=1)

    data_buffer_training_iters = deque(maxlen=20)

    try:
        for i in range(training_iterations):
            # collect self-play data each iteration 100 games
            best_old_model = None
            data_buffer_each = collect_selfplay_data(curr_mcts_player, i, self_play_sizes)
            data_buffer_training_iters.append(data_buffer_each)
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer_training_iters)
            wandb.log({"loss": loss, "entropy": entropy})

            if i == 0:
                """make mcts agent training, eval version"""
                policy_evaluate(env, curr_mcts_player, curr_mcts_player)

                if rl_model == "AC":
                    model_file = f"Training/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(model_file)

                    eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    policy_value_net.save_model(eval_model_file)
                elif rl_model == "QRAC":
                    model_file = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"  # training version train_001
                    policy_value_net.save_model(model_file)

                    eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"  # eval version train_001
                    policy_value_net.save_model(eval_model_file)

                elif rl_model == "EQRAC":
                    assert False, "Not implemented yet"

            else:
                if rl_model == "AC":
                    existing_files = [int(file.split('_')[-1].split('.')[0])
                                      for file in os.listdir(f"Training/{rl_model}_nmcts{n_playout}")
                                      if file.startswith('train_')]
                    old_i = max(existing_files)
                    best_old_model = f"Training/{rl_model}_nmcts{n_playout}/train_{old_i:03d}.pth"

                elif rl_model == "QRAC":
                    existing_files = [int(file.split('_')[-1].split('.')[0])
                                      for file in
                                      os.listdir(f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}")
                                      if file.startswith('train_')]
                    old_i = max(existing_files)
                    best_old_model = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{old_i:03d}.pth"
                elif rl_model == "EQRAC":
                    assert False, "Not implemented yet"
                else:
                    assert False, "don't have model"
                policy_value_net_old = PolicyValueNet(env.state_.shape[1], env.state_.shape[2], quantiles,
                                                      best_old_model, rl_model=rl_model)

                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                win_ratio, curr_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player)

                if (i + 1) % 10 == 0:  # save model 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 (1+10 : total 11)
                    if rl_model == "AC":
                        eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                        policy_value_net.save_model(eval_model_file)
                    elif rl_model == "QRAC":
                        eval_model_file = f"Eval/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                        policy_value_net.save_model(eval_model_file)
                    elif rl_model == "EQRAC":
                        assert False, "Not implemented yet"

                print("Win rate : ", round(win_ratio * 100, 3), "%")
                wandb.log({"Evaluation Win Rate": round(win_ratio * 100, 3)})

                if win_ratio > 0.5:
                    old_mcts_player = curr_mcts_player
                    if rl_model == "AC":
                        model_file = f"Training/{rl_model}_nmcts{n_playout}/train_{i + 1:03d}.pth"
                    elif rl_model == "QRAC":
                        model_file = f"Training/{rl_model}_nmcts{n_playout}_quantiles{quantiles}/train_{i + 1:03d}.pth"
                    elif rl_model == "EQRAC":
                        assert False, "Not implemented yet"
                    else:
                        assert False, "don't have model"

                    policy_value_net.save_model(model_file)
                    print(" ---------- New best policy!!! ---------- ")

                else:
                    # if worse it just reject and does not go back to the old policy
                    print(" ---------- Low win-rate ---------- ")

    except KeyboardInterrupt:
        print('\n\rquit')
