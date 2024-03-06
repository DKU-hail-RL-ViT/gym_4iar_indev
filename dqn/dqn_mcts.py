import numpy as np
import copy
import torch
import random
import argparse


from dqn_network import DQN


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()



def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def policy_value_fn(board, net):
    available = np.where(board[3].flatten() == 0)[0]
    current_state = np.ascontiguousarray(board.reshape(-1, 5, board.shape[1], board.shape[2]))
    log_act_probs, value = net(torch.from_numpy(current_state).float())

    act_probs = np.exp(log_act_probs.data.numpy().flatten())
    filtered_act_probs = [(action, prob) for action, prob in zip(available, act_probs) if action in available]
    state_value = value.item()

    return filtered_act_probs, state_value


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._removed_children = []
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        action, i_node = max(self._children.items(),
                             key=lambda act_node: act_node[1].get_value(c_puct))
        if self._children == {}:
            print('empty')
        return action, i_node

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None

    @property
    def children(self):
        return self._children


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000):

        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, env):  # obs.shape = (5,9,4)

        net = DQN(env.state_.shape[1], env.state_.shape[2])
        node = self._root

        while (1):
            if node.is_leaf():
                break

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, info = env.step(action)

        # print('\t out of while')
        action_probs, leaf_value = policy_value_fn(env.state_, net)
        # print('available:', len(action_probs))

        # Check for end of game
        end, result = env.winner()

        if not end:
            # print("\t node expand")
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if result == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if result == 0 else -1.0
                )
            obs, _ = env.reset()

        node.update_recursive(-leaf_value)

    def get_move_probs(self, env, temp):  # state.shape = (5,9,4)

        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):  # for 400 times
            self._playout(copy.deepcopy(env))  # state_copy.shape = (5,9,4)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.policy_value_fn = policy_value_fn
        self.model = DQN
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, env, temp, return_prob=0):  # env.state_.shape = (5,9,4)
        available = np.where(env.state_[3].flatten() == 0)[0]
        sensible_moves = available

        # Adding batch dimension
        state = torch.tensor(env.state_).unsqueeze(0)  # state.shape : [1, 5, 9, 4]

        self.epsilon *= args.eps_decay  # 여기부터 시작
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]

        if random.random() < self.epsilon:
            print(random.randint(0, self.action_dim - 1))
            # return random.randint(0, self.action_dim - 1)

        print(torch.argmax(q_value).item())
        # return torch.argmax(q_value).item()


        # TODO 말이 안되긴하는데 제대로 오는지 확인하려고 이렇게 해놓았슴

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)  # board.shape = (5,9,4)


            if self._is_selfplay:

                move = torch.argmax(q_value).item()

                # # update the root node and reuse the search tree
                self.mcts.update_with_move(move)

            else:
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")



    def oppo_node_update(self, move):
        self.mcts.update_with_move(move)


    def train(self, max_episodes=1000, render_episodes=100):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state, _ = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state

            if self.buffer.size() >= args.batch_size:
                self.replay()


            if (ep + 1) % render_episodes == 0:
                state, _ = self.env.reset()
                while not done:
                    self.env.render()
                    action = self.model.get_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)

    def __str__(self):
        return "training MCTS {}".format(self.player)



