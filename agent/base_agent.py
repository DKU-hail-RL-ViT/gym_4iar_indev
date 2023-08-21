from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gym_4iar.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from gym_4iar.utils import RunningMeanStats, LinearAnneaer

from gym_4iar.mcts import MCTSPlayer

class BaseAgent(ABC):

    def __init__(self, env, num_steps=5*(10**7), num_actions=36,
                 batch_size=512, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True):

        self.env = env

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

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
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


    def mcts_choose_action(self, state_representation):
        mcts_player = MCTSPlayer()  # MCTSPlayer의 인스턴스 생성
        mcts_player.set_player_ind(0)  # 플레이어 인덱스 설정 (첫 번째 플레이어는 0)
        mcts_player.reset_player()  # 새로운 움직임을 위해 MCTS 트리 리셋

        chosen_action = mcts_player.get_action(state_representation)
        return chosen_action

    def get_state_representation(self):
        # Convert the raw state to a 9 x 4 matrix
        map = np.int16(np.linspace(0, 4 * 9 - 1, 4 * 9)).tolist()
        return map

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        mcts_player = MCTSPlayer()  # Create an instance of the MCTSPlayer
        mcts_player.set_player_ind(0)  # Set the player index (0 for the first player)
        mcts_player.reset_player()  # Reset the MCTS tree for a new move

        # Get the current state representation (convert to the format expected by MCTS)
        state_representation = self.get_state_representation()  # Implement this method

        # Get the chosen action using MCTS
        chosen_action = mcts_player.get_action(state_representation)

        return chosen_action

    def exploit(self, state):
        state_representation = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.

        mcts_player = MCTSPlayer()  # Move this line here to create the MCTSPlayer instance
        mcts_player.set_player_ind(0)
        mcts_player.reset_player()

        with torch.no_grad():
            chosen_action = mcts_player.get_action(state_representation)

        return chosen_action

    @abstractmethod
    def learn(self):
        pass

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better performances.
            self.online_net.sample_noise()

            state_representation = self.get_state_representation()
            action = self.mcts_choose_action(state_representation)
            state_representation.remove(action)

            next_state, reward, done, _ = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if episode_steps > 36:
                break

            self.train_step_interval()

        if (episode_steps % 2 == 0) and (episode_steps <= 36):
            print(f'Episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}     '
                  f'return: {episode_return:<5.1f}  '
                  f'win: white')

        elif (episode_steps % 2 == 1) and (episode_steps <= 36):
            print(f'Episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}     '
                  f'return: {episode_return:<5.1f}  '
                  f'win: black')
        else:
            print(f'Episode: {self.episodes:<4}  '
                  f'win: draw')


    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()
