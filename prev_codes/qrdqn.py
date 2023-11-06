import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np




def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class QRDQNNet(nn.Module):
    def __init__(self, board_width, board_height, num_quantiles):
        super(QRDQNNet, self).__init__()
        self.num_quantiles = num_quantiles
        self.board_width = board_width
        self.board_height = board_height

        # Common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Quantile layers (for each quantile)
        self.quantile_convs = nn.ModuleList(
            [nn.Conv2d(128, 4, kernel_size=1) for _ in range(num_quantiles)])
        self.quantile_fcs = nn.ModuleList(
            [nn.Linear(4 * board_width * board_height, board_width * board_height) for _ in range(num_quantiles)])

    def forward(self, state_input):
        # Common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        quantile_predictions = []
        for i in range(self.num_quantiles):
            # Quantile layers
            x_quantile = F.relu(self.quantile_convs[i](x))
            x_quantile = x_quantile.view(-1, 4 * self.board_width * self.board_height)
            x_quantile = F.log_softmax(self.quantile_fcs[i](x_quantile), dim=1)
            quantile_predictions.append(x_quantile)

        return quantile_predictions

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        current_state = torch.from_numpy(current_state).float()

        if self.use_gpu:
            current_state = current_state.cuda()

        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = torch.exp(log_act_probs).cpu().detach().numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.item()
        return act_probs, value



class QRDQN:
    def __init__(self, board_width, board_height, num_actions, num_quantiles, lr=0.001):
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qrdqn_net = QRDQNNet(board_width, board_height, num_quantiles).to(self.device)
        self.optimizer = optim.Adam(self.qrdqn_net.parameters(), lr=lr)



    def train(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute QR-DQN target values
        with torch.no_grad():
            next_state_quantiles = self.qrdqn_net(next_state_batch)
            next_action = next_state_quantiles[0].mean(dim=2).argmax(dim=1)
            target_quantiles = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * next_state_quantiles[
                torch.arange(len(next_action)), next_action]

        current_quantiles = self.qrdqn_net(state_batch)
        selected_quantiles = current_quantiles[torch.arange(len(action_batch)), action_batch]

        # Compute Quantile Huber Loss
        td_errors = target_quantiles.unsqueeze(2) - selected_quantiles.unsqueeze(1)
        huber_loss = self.huber_loss(td_errors)
        loss = torch.abs(self.tau - (td_errors < 0).float()) * huber_loss
        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state, epsilon=0.1):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            quantiles = self.qrdqn_net(state)[0]
            if np.random.rand() < epsilon:
                action = np.random.randint(self.num_actions)
            else:
                action = quantiles.mean(dim=2).argmax().item()
        return action

    def huber_loss(self, x, delta=1.0):
        return torch.where(x.abs() < delta, 0.5 * x.pow(2), delta * (x.abs() - 0.5 * delta))

    def set_tau(self, tau):
        self.tau = tau

##########################


class PolicyValueNet():
    """Policy-value network"""
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False, num_quantiles=32):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.num_quantiles = num_quantiles
        self.board_height = board_height
        self.l2_const = 1e-4  # Coefficient of L2 penalty

        # The policy value net module
        if self.use_gpu:
                self.policy_value_net = QRDQNNet(board_width, board_height, num_quantiles).cuda()
        else:
            self.policy_value_net = QRDQNNet(board_width, board_height, num_quantiles)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file, map_location=torch.device('cuda' if self.use_gpu else 'cpu'))
            self.policy_value_net.load_state_dict(net_params)



    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = np.array(state_batch, dtype=np.float32)
        state_batch = torch.FloatTensor(state_batch)

        if self.use_gpu:
            state_batch = state_batch.cuda()

        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = torch.exp(log_act_probs)
        return act_probs.cpu().detach().numpy(), value.cpu().detach().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        current_state = torch.from_numpy(current_state).float()

        if self.use_gpu:
            current_state = current_state.cuda()

        log_act_probs = self.policy_value_net(current_state)
        act_probs = torch.exp(log_act_probs).cpu().detach().numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs

    def train_step(self, state_batch, mcts_probs, quantiles_batch, lr):
        """Perform a training step"""
        state_batch = np.array(state_batch, dtype=np.float32)
        mcts_probs = np.array(mcts_probs, dtype=np.float32)
        quantiles_batch = np.array(quantiles_batch, dtype=np.float32)

        state_batch = torch.FloatTensor(state_batch)
        mcts_probs = torch.FloatTensor(mcts_probs)
        quantiles_batch = torch.FloatTensor(quantiles_batch)

        if self.use_gpu:
            state_batch = state_batch.cuda()
            mcts_probs = mcts_probs.cuda()
            quantiles_batch = quantiles_batch.cuda()

        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Set learning rate
        set_learning_rate(self.optimizer, lr)

        # Forward pass
        quantile_predictions = self.policy_value_net(state_batch)
        # Define the loss using quantile Huber loss
        loss = quantile_huber_loss(quantile_predictions, quantiles_batch)
        # Backward and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """Save model params to file"""
        net_params = self.get_policy_param()  # Get model params
        torch.save(net_params, model_file)
