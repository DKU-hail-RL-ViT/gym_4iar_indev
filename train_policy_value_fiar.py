from fiar_env import Fiar, turn, action2d_ize
import numpy as np
import wandb
import random

from collections import defaultdict, deque
from mcts import MCTSPlayer
from policy_value_network import PolicyValueNet

eps = 0.05

batch_size = 512
temp = 1e-3
learn_rate = 2e-3
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL

n_playout = 400  # num of simulations for each move
c_puct = 5
buffer_size = 10000
epochs = 5  # num of train_steps for each update
kl_targ = 0.02
check_freq = 50
self_play_times = 1500
best_win_ratio = 0.0


def collect_selfplay_data(n_games=1):
    for i in range(n_games):
        rewards, play_data = self_play(env, temp=temp)
        play_data = list(play_data)[:]
        data_buffer.extend(play_data)


def self_play(env, temp=1e-3):
    states, mcts_probs, current_player = [], [], []
    obs, _ = env.reset()

    player_0 = turn(obs)
    player_1 = 1 - player_0

    obs_post[0] = obs[player_0]
    obs_post[1] = obs[player_1]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[player_0] + obs[player_1]

    while True:
        while True:
            action = None
            move_probs = None

            if obs[3].sum() == 36:
                print('draw')
            else:
                move, move_probs = mcts_player.get_action(env, obs_post, temp=temp, return_prob=1)
                action = move

            action2d = action2d_ize(action)
            if obs[3, action2d[0], action2d[1]] == 0.0:
                break

        # store the data
        states.append(obs)
        mcts_probs.append(move_probs)
        current_player.append(turn(obs))

        print(player_0, player_1)
        obs, reward, terminated, info = env.step(action)

        player_0 = turn(obs)
        player_1 = 1 - player_0

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.winner()

        if end:  # 이 부분 수정해야 함
            if obs[3].sum() == 36:
                print('draw')
                env.render()  # ?

            winners_z = np.zeros(len(current_player))

            if winners != 0:
                winners_z[np.array(current_player) == winners] = 1.0
                winners_z[np.array(current_player) != winners] = -1.0

            obs, _ = env.reset()

            return reward, zip(states, mcts_probs, winners_z)


def policy_update(lr_multiplier=1.0):
    print('80%')

    """update the policy-value net"""
    mini_batch = random.sample(data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)

    for i in range(epochs):
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

    explained_var_old = (1 -
                         np.var(np.array(winner_batch) - old_v.flatten()) /
                         np.var(np.array(winner_batch)))
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy


def policy_evaluate():
    win_ratio = 100
    return win_ratio


if __name__ == '__main__':

    wandb.init(project="4iar_DQN")

    env = Fiar()
    obs, _ = env.reset()
    data_buffer = deque(maxlen=buffer_size)
    policy_value_net = PolicyValueNet(obs.shape[1], obs.shape[2])

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    # obs_post = obs[player_myself] + obs[player_enemy]*(-1)

    c = 0
    num_timesteps = 0
    ep = 1
    b_win = 0
    w_win = 0
    self_play_sizes = 1

    mcts_player = MCTSPlayer(c_puct, n_playout, is_selfplay=1)

    for i in range(self_play_times):
        collect_selfplay_data(self_play_sizes)

        if len(data_buffer) > batch_size:
            loss, entropy = policy_update()

        if (i + 1) % check_freq == 0:
            print("current self-play batch: {}".format(i + 1))
            win_ratio = policy_evaluate()
            print(win_ratio)

            """
            else:
                if player_0 == 0.0:
                    b_win += 1
                elif player_0 == 1.0:
                    w_win += 1

            b_wins = b_win / ep
            w_wins = w_win / ep

            print({"episode ": ep, "black win (%)": round(b_wins, 5) * 100, "white win (%)": round(w_wins, 5) * 100,
                  "black wins time": b_win,"white wins time": w_win, "tie time": ep - b_win - w_win})
            print('\n\n')
            # 나중에 이부분 round로 나두지말고 format으로 처리해서 부동소수점 문제 처리

            c = 0
            ep += 1

            # evaluation against random agent
            # if ep % 1000 == 0:
            #    rewards, wons = evaluation_against_random(env, model)
                # save model
            #    model.save("qrdqn_fiar")

            """