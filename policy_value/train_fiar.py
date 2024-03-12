import numpy as np
import random
import os

from collections import defaultdict, deque
from fiar_env import Fiar, turn, action2d_ize
from policy_value_network import PolicyValueNet
from policy_value.mcts import MCTSPlayer

# from policy_value.policy_value_mcts_pure import RandomAction


""" tuning parameter """
n_playout = 10  # = MCTS simulations(n_mcts) & training 2, 20, 50, 100, 400
check_freq = 2  # = more self_playing & training 1, 10, 20, 50, 100

""" MCTS parameter """
buffer_size = 1000
c_puct = 5
epochs = 10  # During each training iteration, the DNN is trained for 10 epochs.
self_play_sizes = 1
game_batch_num = check_freq * 100
temperature = 0.1

""" Policy update parameter """
batch_size = 64  # previous 512
learn_rate = 2e-4  # previous 2e-3
lr_mul = 1.0
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
kl_targ = 0.02  # previous 0.02

""" Policy evaluate parameter """
win_ratio = 0.0
init_model = None

""" RL name """
rl_model = "AC"
# rl_model = "DQN"
# rl_model = "QRDQN"


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


def collect_selfplay_data(mcts_player, temp, n_games=100):
    # self-play 100 games and save in data_buffer(queue)

    for i in range(n_games):
        rewards, play_data = self_play(env, mcts_player, temp=temp)
        play_data = list(play_data)[:]

        play_data = get_equi_data(env, play_data)
        data_buffer.extend(play_data)
        # if i >= (n_games - last_n_games):
        #    play_data = get_equi_data(env, play_data)
        #    data_buffer.extend(play_data)


def self_play(env, mcts_player, temp):
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


def policy_update(lr_mul, policy_value_net):
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
    return loss, entropy, lr_multiplier, policy_value_net


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    # leaf_mcts_player = MCTS_leaf(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    # random_action_player = RandomAction() # random actions Agent
    win_cnt = defaultdict(int)

    for j in range(n_games):

        # reset for each game
        winner = start_play(env,
                            training_mcts_player,
                            opponent_mcts_player,
                            temperature)
        if winner == -0.5:
            winner = 0
        win_cnt[winner] += 1
        print("{} / 30 ".format(j + 1))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[0], win_cnt[-1]))
    return win_ratio, training_mcts_player


def start_play(env, player1, player2, temp):
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
        move = player_in_turn.get_action(env, temp)
        # print(move)
        obs, reward, terminated, info = env.step(move)
        assert env.state_[3][action2d_ize(move)] == 1, ("Invalid move", action2d_ize(move))
        end, winner = env.winner()

        if not end:
            # print("\t opponent_update")
            current_player = 1 - current_player
            player_in_turn = players[current_player]
            player_in_turn.oppo_node_update(move)

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
                                          model_file=init_model, rl_model=rl_model)
    else:
        # start training from a new policy-value net
        policy_value_net = PolicyValueNet(env.state().shape[1],
                                          env.state().shape[2], rl_model=rl_model)

    # policy_value_net_old = copy.deepcopy(policy_value_net)
    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)

    try:
        for i in range(game_batch_num):
            """ During the first 15 steps of each self-play game,
            the temperature is set to 1 to induce variability in the data """
            temperature = 1 if i < 15 * check_freq else 0.1
            collect_selfplay_data(curr_mcts_player, temperature, self_play_sizes) # TODO mcts가 playout 할때마다 새로 Net 선언함

            if len(data_buffer) > batch_size:
                loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                               policy_value_net=policy_value_net)
                print(policy_value_net)
                print('\t update 직후 policy_value_net')
                # wandb.log({"loss": loss, "entropy": entropy})

            if (i + 1) % check_freq == 0:
                # When MCTS evaluate level self play turn off
                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)

                if (i + 1) == check_freq:
                    policy_evaluate(env, curr_mcts_player, curr_mcts_player)
                    model_file = f"nmcts{n_playout}_iter{check_freq}/train_{i + 1:05d}.pth"
                    policy_value_net.save_model(model_file)

                    # meaning it is the first one and never saved any
                    policy_value_net_old = policy_value_net
                    old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                    print("sdf")

                else:
                    existing_files = [int(file.split('_')[-1].split('.')[0])
                                      for file in os.listdir(f"nmcts{n_playout}_iter{check_freq}")
                                      if file.startswith('train_')]
                    old_i = max(existing_files) if existing_files else check_freq
                    best_old_model = f"nmcts{n_playout}_iter{check_freq}/train_{old_i:05d}.pth" # 이것도 하나로 줄일수 있을거 같은데

                    policy_value_net_old = PolicyValueNet(env.state_.shape[1], env.state_.shape[2], best_old_model)
                    old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)

                    win_ratio, best_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player)
                    print("\t win rate : ", win_ratio * 100, "%")

                    if win_ratio > 0.5:
                        curr_mcts_player = best_mcts_player


                        model_file = f"nmcts{n_playout}_iter{check_freq}/train_{i + 1:05d}.pth"
                        policy_value_net.save_model(model_file)
                        # best_mcts_player.policy_value_fn.save_model(model_file)
                        print("\t New best policy!!!")

                        print("\t 자 policy_value_net이 바뀌나요?")
                        print(policy_value_net)

                        # curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                        # policy_value_net_old = copy.deepcopy(policy_value_net)
                        print("\t high win-rate 60%")

                    else:  # if worse it just reject and does not go back
                        print("\t low win-rate 60%")

            else:
                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)
                print("\t no check-point")
                print(policy_value_net)
    except KeyboardInterrupt:
        print('\n\rquit')
