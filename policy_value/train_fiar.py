import numpy as np
import wandb
import random
import os

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from policy_value.mcts import MCTSPlayer
from policy_value.mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_network import PolicyValueNet
# from policy_value.policy_value_mcts_pure import RandomAction

# fine-tuning models
n_playout = 2  # = MCTS simulations(n_mcts) & training 2, 20, 50, 100, 400
check_freq = 1  # = iter & training 1, 10, 20, 50, 100

# num of simulations for each move
self_play_sizes = 1


buffer_size = 10000
c_puct = 5
epochs = 1  # During each training iteration, the DNN is trained for 10 epochs.
self_play_times = 100 # 비교할 논문에서는 100번 했다고 함. # previous 1500
temp = 0.1

# policy update parameter
batch_size = 64  # previous 512
learn_rate = 2e-4   # previous 2e-3
lr_mul = 1.0
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
best_win_ratio = 0.0
pure_mcts_playout_num = 2  # [todo] 디버깅하려고 줄여놓음 previous 500

win_ratio = 0.0
kl_targ = 0.02  # previous 0.02

init_model = None



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


def collect_selfplay_data(n_games=30):      #[Todo] 이부분 수정 해야함
    last_n_games = 20
    for i in range(n_games):
        # temp = 1 if i <= 15 else 0.1
        rewards, play_data = self_play(env, temp=temp)
        play_data = list(play_data)[:]

        if i >= (n_games - last_n_games):
            play_data = get_equi_data(env, play_data)
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
                print('self_play_draw')
            else:
                move, move_probs = mcts_player.get_action(env, temp=temp, return_prob=1)
                action = move
            action2d = action2d_ize(action)

            if obs[3, action2d[0], action2d[1]] == 0.0:
                break

        # store the data
        states.append(obs)
        mcts_probs.append(move_probs)
        current_player.append(turn(obs))

        obs, reward, terminated, info = env.step(action)

        player_0 = turn(obs)
        player_1 = 1 - player_0

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        end, winners = env.winner()

        if end:
            if obs[3].sum() == 36:
                print('self_play_draw')

            print(env)
            obs, _ = env.reset()

            # reset MCTS root node
            mcts_player.reset_player()

            print("batch i:{}, episode_len:{}".format(
                i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if winners != -1:
                if winners == -0.5:  # if win white return : 0.1
                    winners = 0
                winners_z[np.array(current_player) == 1 - winners] = 1.0
                winners_z[np.array(current_player) != 1 - winners] = -1.0
            return winners, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul):
    kl, loss, entropy = 0, 0, 0
    lr_multiplier = lr_mul

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
                         (np.var(np.array(winner_batch)) + 1e-10))
    explained_var_new = (1 -
                         np.var(np.array(winner_batch) - new_v.flatten()) /
                         (np.var(np.array(winner_batch)) + 1e-10))

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
    return loss, entropy, lr_multiplier


def policy_evaluate(env, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(policy_value_fn, c_puct=c_puct, n_playout=n_playout)   # training Agent
    pure_mcts_player = MCTSPlayer(policy_value_fn, c_puct=c_puct, n_playout=n_playout) # first evaluate Agent
    # random_action_player = RandomAction() # random actions Agent
    win_cnt = defaultdict(int)

    for j in range(n_games):
        winner = start_play(env,
                            current_mcts_player,
                            pure_mcts_player)

        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio


def policy_evaluate2(env, n_games=30):
    existing_files = [int(file.split('_')[-1].split('.')[0])
                      for file in os.listdir('nmcts2_iter50')
                      if file.startswith('pure_mcts_')]
    max_i = max(existing_files) if existing_files else 0

    best_model_file = 'nmcts2_iter50/pure_mcts_{}.pth'.format(max_i)
    best_policy = PolicyValueNet(env.state_.shape[1], env.state_.shape[2], best_model_file)
    prev_best_player = MCTSPlayer(best_policy.policy_value_fn,
                                  c_puct=5,
                                  n_playout=2)    # previous 400
    win_cnt = defaultdict(int)
    for j in range(n_games):
        winner = start_play(env,
                            mcts_player,
                            prev_best_player)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1    # white win
        print("{} / 30 ".format(j + 1))

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("win: {}, lose: {}, tie:{}".format(
            win_cnt[1], win_cnt[0], win_cnt[-1]))

    print("win rate : ", win_ratio * 100, "%")
    return win_ratio


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

    while True:
        player_in_turn = players[current_player]

        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env) # TODO
        print(move)
        obs, reward, terminated, info = env.step(move)
        print(env.state_[3][action2d_ize(move)])
        print(action2d_ize(move))
        assert env.state_[3][action2d_ize(move)]==1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            current_player = 1 - current_player
            # player_in_turn = players[current_player]

            # if move is not None:
            #     player_in_turn.mcts
            #     player_in_turn.mcts._root.children.pop(move)  # pop the prev player's move
            # player_in_turn.node_update(move)
            # [Todo] 이 부분에서 안끝났다면 상대방의 _root_children을 없애러
        else:
            print(env)
            obs, _ = env.reset()

            return winner


if __name__ == '__main__':

    # wandb.init(mode="offline",
    #           entity="hails",
    #           project="policy_value_4iar")

    env = Fiar()
    obs, _ = env.reset()
    data_buffer = deque(maxlen=buffer_size)

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    if init_model:
        # start training from an initial policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2],
                                          model_file=init_model)
    else:
        # start training from a new policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2])

    mcts_player = MCTSPlayer(policy_value_fn, c_puct, n_playout, is_selfplay=1)

    try:
        for i in range(self_play_times):
            collect_selfplay_data(self_play_sizes)

            if len(data_buffer) > batch_size:
                loss, entropy, lr_multiplier = policy_update(lr_mul=lr_multiplier)
                # wandb.log({"loss": loss, "entropy": entropy})

            if (i + 1) % check_freq == 0:
                print("current self-play batch: {}".format(i + 1))

                if (i + 1) == check_freq:
                    win_ratio = policy_evaluate(env)
                    print("win rate : ", win_ratio * 100, "%")

                    model_file = 'nmcts2_iter50/pure_mcts_{}.pth'.format(i + 1)
                    policy_value_net.save_model(model_file)
                    best_win_ratio = win_ratio

                # ############### AI VS AI ###################
                # load the trained policy_value_net

                else:
                    # obs_post = np.zeros((5, 9, 4))
                    win_ratio = policy_evaluate2(env)
                    print("win rate : ", win_ratio * 100, "%")

                    if win_ratio > best_win_ratio:  # update the best_policy
                        print("New best policy!!!!!!!!")
                        best_win_ratio = win_ratio
                        model_file = 'nmcts2_iter50/pure_mcts_{}.pth'.format(i + 1)
                        policy_value_net.save_model(model_file)

            if i > self_play_times:    # Save up to 100 models
                print("End loop")
                break

    except KeyboardInterrupt:
        print('\n\rquit')
