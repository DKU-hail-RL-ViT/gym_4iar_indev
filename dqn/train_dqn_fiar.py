import numpy as np
import wandb

from Project.fiar_env import Fiar, turn, action2d_ize
from collections import deque
from Project.dqn.dqn_mcts import MCTSPlayer
from Project.dqn.dqn_mcts_pure import MCTSPlayer as MCTS_Pure
from Project.model.dqn import DQN


# self-play parameter
c_puct = 5
n_playout = 400  # previous 400
# num of simulations for each move

self_play_sizes = 1
temp = 1e-3

epochs = 5  # num of train_steps for each update
self_play_times = 1000   # previous 1500
pure_mcts_playout_num = 500     # previous 1000

# policy update parameter
batch_size = 64  # previous 512
learn_rate = 2e-3
lr_mul = 1.0
lr_multiplier = 1.0     # adaptively adjust the learning rate based on KL
check_freq = 1  # previous 50
best_win_ratio = 0.0

kl_targ = 0.02  # previous 0.02


# dqn parameter
buffer_size = 10000


total_timesteps = 100000
learning_starts = 1000
eps = 0.05


init_model = None


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = np.random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)


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

            if obs[3, action2d[0], action2d[1]] == 0:
                break

        # [Todo] 지난번처럼 copy한 state랑 현재 state가 다르게 들어갈수도
        states.append(obs_post.copy())
        mcts_probs.append(move_probs)
        current_player.append(player_0)

        print(action)

        obs, reward, terminated, info = env.step(action)

        player_0 = turn(obs)
        player_1 = 1 - player_0

        obs_post[0] = obs[player_0]
        obs_post[1] = obs[player_1]
        obs_post[2] = np.zeros_like(obs[0])
        obs_post[3] = obs[player_0] + obs[player_1]

        next_states = obs_post.copy()
        replay_buffer.add(obs_post, action, reward, next_states, terminated)



        end, winners = env.winner()

        if end:
            if obs[3].sum() == 36:
                print('draw')

            print(env)
            obs, _ = env.reset()

            # reset MCTS root node
            mcts_player.reset_player()
            print("batch i:{}, episode_len:{}".format(
                i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                model.train(batch)
                # model.train(batch_size=model.batch_size, gradient_steps=1)

            if winners != -1:
                if winners == -0.5:  # if win white return : 0.1
                    winners = 0
                winners_z[np.array(current_player) == 1 - winners] = 1.0
                winners_z[np.array(current_player) != 1 - winners] = -1.0
            return reward, zip(states, mcts_probs, winners_z)



def policy_update():
    model._store_transition(model.replay_buffer, np.array([action]), obs_post.reshape(*[1, *obs_post.shape]),
                            np.array([reward]), np.array([terminated]), [info])










if __name__ == '__main__':

    wandb.init(mode="offline",
               entity="hails",
               project="4iar_DQN")

    env = Fiar()
    obs, _ = env.reset()
    data_buffer = deque(maxlen=buffer_size)

    model = DQN("MlpPolicy", env, verbose=1, learning_starts=learning_starts)

    total_timesteps, callback = model._setup_learn(
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="DQN_fiar",
        progress_bar=True,
    )

    turn_A = turn(obs)
    turn_B = 1 - turn_A

    obs_post = obs.copy()
    obs_post[0] = obs[turn_A]
    obs_post[1] = obs[turn_B]
    obs_post[2] = np.zeros_like(obs[0])
    obs_post[3] = obs[turn_A] + obs[turn_B]

    mcts_player = MCTSPlayer(c_puct, n_playout)
    replay_buffer = ReplayBuffer(buffer_size)

    try:
        for i in range(self_play_times):
            collect_selfplay_data(self_play_sizes)
            print("1")

        if len(data_buffer) > batch_size:
            # data_buffer.append((rewards, wons))
            a = policy_update()

        # env.reset()





        # 그 다음에 평가하는 부분
        while True:
            # sample an action until a valid action is sampled
            while True:
                if obs[3].sum() == 36:
                    print('draw')
                    break
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    if player_0 == 0:  # black train version
                        action = model.predict(obs_post.reshape(*[1, *obs_post.shape]))[0]
                        action = action[0]
                    else:
                        action = env.action_space.sample()

                # action = env.action_space.sample()
                action2d = action2d_ize(action)

                if obs[3, action2d[0], action2d[1]] == 0:
                    break

            player_0 = turn(obs)
            player_1 = 1 - player_0

            obs, reward, terminated, info = env.step(action)



            model._store_transition(model.replay_buffer, np.array([action]), obs_post.reshape(*[1, *obs_post.shape]),
                                    np.array([reward]), np.array([terminated]), [info])

            # if num_timesteps > 0 and num_timesteps > learning_starts:
            #     model.train(batch_size=model.batch_size, gradient_steps=1)

            if terminated:
                print("1234123")

    except KeyboardInterrupt:
        print('\n\rquit')


