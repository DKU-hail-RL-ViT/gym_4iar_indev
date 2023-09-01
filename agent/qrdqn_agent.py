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

        print(self.N_index-1)




        # Online network.
        self.online_net = QRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=k).to(self.device)
        # Target network.
        self.target_net = QRDQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=k).to(self.device)

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

    def learn(self):
        self.learning_steps += 1

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights = \
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size)
            weights = None

        quantile_loss, mean_q, errors = self.calculate_loss(
            states, actions, rewards, next_states, dones, weights)

        assert errors.shape == (self.batch_size, 1)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.use_per:
            self.memory.update_priority(errors)

    def calculate_loss(self, states, actions, rewards, next_states, dones, weights):

        smoothing = 0.005
        theta = 0.0005  # hyperparameter
        k = self.N[self.N_index-1]
        i = 0

        while (smoothing > theta) and (k <= 64):
            print(k, "quantile이 업데이트 되었는지 출력")

            # Calculate quantile values of current states and actions at taus.
            current_sa_quantiles = evaluate_quantile_at_action(self.online_net(states=states), actions, i)
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

            print(k, "업데이트한 파라미터를 사용하였는지 출력")
            print(smoothing)
            k *= 2
            i += 1

        # Calculate quantile values of current states and actions at taus.
        self.k = k / 2

        next_actions = torch.argmax(next_q, dim=1, keepdim=True)
        assert next_actions.shape == (self.batch_size, 1)

        # Calculate quantile values of next states and actions at tau_hats.
        next_sa_quantiles = evaluate_quantile_at_action(
            self.target_net(states=next_states),
            next_actions, i).transpose(1, 2)
        assert next_sa_quantiles.shape == (self.batch_size, 1, self.k)

        # Calculate target quantile values.
        target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
        assert target_sa_quantiles.shape == (self.batch_size, 1, self.k)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.k, self.k)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, self.tau_hats, weights, self.kappa)

        return quantile_huber_loss, next_q.detach().mean().item(), \
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)
