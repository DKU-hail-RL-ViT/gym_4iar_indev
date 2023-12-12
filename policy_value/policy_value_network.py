import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.flatten = nn.Flatten()

        # common layers
        self.fc1 = nn.Linear(5*board_width*board_height, 256)
        self.fc2 = nn.Linear(256, 128)

        # action policy layers (MLP)
        self.act_fc1 = nn.Linear(128, board_width*board_height)

        # state value layers (MLP)
        self.val_fc1 = nn.Linear(128, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # flatten input
        x = self.flatten(x)

        # common layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # action policy layers
        action_probs = F.log_softmax(self.act_fc1(x), dim=1)

        # state value layers
        x_val = F.relu(self.val_fc1(x))
        state_value = torch.tanh(self.val_fc2(x_val)).squeeze()  # Fix here

        return action_probs, state_value


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width      # 9
        self.board_height = board_height    # 4
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch_np = np.array(state_batch)
            state_batch = torch.FloatTensor(state_batch_np).cuda()
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.item()
        else:
            state_batch_np = np.array(state_batch)
            state_batch = torch.FloatTensor(state_batch_np)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.detach().cpu().numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Tensor
        state_batch = torch.FloatTensor(np.array(state_batch))
        mcts_probs = torch.FloatTensor(np.array(mcts_probs))
        winner_batch = torch.FloatTensor(np.array(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
