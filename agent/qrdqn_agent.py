import torch
from torch.optim import Adam

from gym_4iar.model import QRDQN
from gym_4iar.utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params
from gym_4iar.agent.base_agent import BaseAgent


class QRDQNAgent(BaseAgent):
    def __init__(self, env, num_steps=5 * (10 ** 7), num_actions=36,
                 batch_size=512, kappa=1.0, lr=5e-5, memory_size=10 ** 6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, use_per=False,
                 max_episode_steps=27000, grad_cliping=None, cuda=True):

        super(QRDQNAgent, self).__init__(
            env, num_steps, num_actions, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            use_per, max_episode_steps, grad_cliping, cuda)

        self.env = env

        self.N = [2, 4, 8, 16, 32, 64]
        self.N_index = 0

        if self.N[self.N_index] == 2:
            self.N_index += 1
            k = self.N[self.N_index - 1]

        else:
            self.N_index += 1
            k = self.N[self.N_index - 1]

        # Online network.
        self.online_net = QRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N_index=self.N_index).to(self.device)

        # Target network.
        self.target_net = QRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N_index=self.N_index).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()

        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2 / batch_size)

        # Fixed fractions.
        taus = torch.arange(
            0, k + 1, device=self.device, dtype=torch.float32) / k
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, k)

        self.kappa = kappa


    def online_net_forward(self, states):
        return self.online_net(states)

    def learn(self):

        smoothing = 0.005
        theta = 0.0005  # hyperparameter
        k = self.N[0]
        i = 0
        self.learning_steps += 1

        while (smoothing > theta) and (k <= 64):

            taus = torch.arange(
                0, k + 1, device=self.device, dtype=torch.float32) / k
            self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, k)

            if self.use_per:
                (states, actions, rewards, next_states, dones), weights = \
                    self.memory.sample(self.batch_size)
            else:
                states, actions, rewards, next_states, dones = \
                    self.memory.sample(self.batch_size)
                weights = None

            print(k, "agent에서 quantile이 업데이트 되었는지 출력")

            print(self.tau_hats, "tau")

            # Calculate quantile values of current states and actions at taus.
            current_sa_quantiles = evaluate_quantile_at_action(self.online_net_forward(states=states), actions)
            assert current_sa_quantiles.shape == (self.batch_size, k, 1)

            with torch.no_grad():
                # Calculate Q values of next states.
                next_q = self.target_net.calculate_q(states=next_states)

                # Calculate greedy actions.
                next_q_sorted, next_q_indices = torch.sort(next_q, descending=True, dim=1)
                max_q_values = next_q_sorted[:, 0]
                second_max_q_values = next_q_sorted[:, 1]

                diff = abs(max_q_values - second_max_q_values)
                smoothing = torch.mean(diff)

            print(k, "qrdqn 에이전트에서 업데이트한 파라미터를 사용하였는지 출력")
            print(smoothing)
            k *= 2
            i += 1

        # Calculate quantile loss
        td_errors = self.calculate_td_errors(states, actions, rewards, next_states, dones, k)
        quantile_loss = calculate_quantile_huber_loss(
            td_errors, self.tau_hats, weights, self.kappa)

        assert td_errors.shape == (self.batch_size, self.k, self.k)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        # if self.use_per:
        #     self.memory.update_priority(errors)

    def calculate_td_errors(self, states, actions, rewards, next_states, dones, k):
        # Calculate quantile values of current states and actions at taus.
        current_sa_quantiles = evaluate_quantile_at_action(self.online_net(states), actions)
        assert current_sa_quantiles.shape == (self.batch_size, k, 1)

        # Calculate Q values of next states.
        next_q = self.target_net.calculate_q(states=next_states)

        # Calculate greedy actions.
        next_q_sorted, next_q_indices = torch.sort(next_q, descending=True, dim=1)
        max_q_values = next_q_sorted[:, 0]
        second_max_q_values = next_q_sorted[:, 1]

        diff = abs(max_q_values - second_max_q_values)
        smoothing = torch.mean(diff)

        k *= 2

        # Calculate quantile loss
        td_errors = self.calculate_td_errors(states, actions, rewards, next_states, dones, k)
        quantile_loss = calculate_quantile_huber_loss(
            td_errors, self.tau_hats, weights, self.kappa)

        return td_errors
