from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import wandb

from gym_4iar.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from gym_4iar.qrdqn_utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, num_steps=5*(10**7), num_actions=36,
                 batch_size=512, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000,  use_per=False,
                 eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=40, grad_cliping=5.0, cuda=True):

        self.env = env
        self.map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9)).tolist()

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step, beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.b_win = 0
        self.w_win = 0

        self.use_per = use_per

        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping
        self.play_times = 100



    def run(self):
        for i in range(self.play_times):
            # self.collect_selfplay(self.play_times)
            self.train_episode(self_play=True)

        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    """

    def collect_selfplay(self, n_games=1):

        for i in range(n_games):
            winner, play_data = self.selfplay(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    
    def selfplay(self, player,  temp=1e-3):
       
        self.env.reset()
        p1, p2 = self.env.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)

            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()

                return winner, zip(states, mcts_probs, winners_z)

    """














    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self, episode_map):
        if len(episode_map) == 0:
            # If all actions have been taken, reset the map.
            episode_map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9)).tolist()

        action = np.random.choice(episode_map)
        episode_map.remove(action)

        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.online_net.calculate_q(states=state).argmax().item()
        return action

    @abstractmethod
    def learn(self):
        pass

    def train_episode(self, self_play=False):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()
        episode_map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9)).tolist()

        memory = self.self_play_memory if self_play else self.memory

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better performances.
            self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore(episode_map)
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)

            if self_play:
                # Save the self-play experience to memory
                self.memory.append(state, action, reward, next_state, done)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if episode_steps > 36:
                break

            self.train_step_interval()

        if episode_return == 1.0:
            self.b_win += 1
        elif episode_return == -1.0:
            self.w_win += 1

        b_win = self.b_win / self.episodes
        w_win = self.w_win / self.episodes

        if (episode_steps % 2 == 0) and (episode_steps <= 36):
            wandb.log({"Episode ": self.episodes, "episode steps": episode_steps,
                       "black win (%)": b_win * 100, "white win (%)": w_win * 100})
        elif (episode_steps % 2 == 1) and (episode_steps <= 36):
            wandb.log({"Episode ": self.episodes, "episode steps": episode_steps,
                       "black win (%)": b_win * 100, "white win (%)": w_win * 100})
        else:
            wandb.log({"Episode ": self.episodes, "episode steps": episode_steps - 1,
                       "black win (%)": b_win * 100, "white win (%)": w_win * 100})


    def train_step_interval(self):
            self.epsilon_train.step()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.is_update():
                self.learn()