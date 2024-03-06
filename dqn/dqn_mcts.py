import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse

from dqn.policy_value_network import DQN
from collections import deque


parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# TODO 음..
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
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, env):  # obs.shape = (5,9,4)
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        # print('\t init playout')
        dqn_shape = np.reshape(env.state_, (1,) + env.state_.shape)
        net = DQN(env.state_.shape[1], env.state_.shape[2], dqn_shape, env.state_.shape[1]*env.state_.shape[2])
        node = self._root

        while (1):
            if node.is_leaf():
                break

            # Greedily select next move.
            # TODO  Here q_value to get action
            obs, reward, terminated, info = env.step(action)

        # print('\t out of while')
        # Todo dqn 방식에서 leaf_value data는 우째 가져오지, 위의 코드 자체가 안돌아가는데
        action_probs, leaf_value = policy_value_fn(env.state_, net)

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



    # def get_move_probs(self, env, temp):  # state.shape = (5,9,4)

        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        #  for n in range(self._n_playout):
        #     self._playout(copy.deepcopy(env))  # state_copy.shape = (5,9,4)

        # calc the move probabilities based on visit counts at the root node
        # act_visits = [(act, node._n_visits)
        #               for act, node in self._root._children.items()]

        # acts, visits = zip(*act_visits)
        #act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        # return acts, act_probs

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
        self._is_selfplay = is_selfplay
        self.n_playout = n_playout

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)


    # TODO 여기도 수정
    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values.numpy()


    # TODO 여기도 수정
    def get_action(self, env, temp, return_prob=0):  # env.state_.shape = (5,9,4)
        available = np.where(env.state_[3].flatten() == 0)[0]
        sensible_moves = available
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])

        if len(sensible_moves) > 0:
            for n in range(self.mcts._n_playout):
                self.mcts._playout(copy.deepcopy(env))
            # acts, probs = self.mcts.get_move_probs(env, temp)
            # board.shape = (5,9,4)
            # move_probs[list(acts)] = probs

            if self._is_selfplay:
                # 항상 q_value를 반환해야할거 같은데
                state = np.reshape(env.state_, (1,) + env.state_.shape)  # state.shape (1, 5, 9, 4)

                self.epsilon *= args.eps_decay
                self.epsilon = max(self.epsilon, args.eps_min)
                q_value = self.predict(state)[0]
                if np.random.random() < self.epsilon:
                    # return np.random.choice(self.action_dim)
                    print(np.random.choice(sensible_moves))
                print(np.argmax(q_value))
                    # return np.random.choice(sensible_moves) # TODO 여기 수정해야함 주석처리한게 원래 코드
                # return np.argmax(q_value)

                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)

            else:

                move = np.random.choice(acts, p=probs)
                print("my name")
                # reset the root node
                assert len(np.where(np.abs(env.state_[3].reshape((-1,))-1 ))[0]) == len(self.mcts._root.children)
                self.mcts.update_with_move(-1)

            if return_prob:  # TODO 어차피 probs반환 못하니까 move만 return 하면 됨
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def oppo_node_update(self, move):
        # TODO 이 줄이 selfplay 중에도 하다보니 문제가 생길 수 있지 않을까 하는 생각
        # 원래는 없없던 코드
        self.mcts.update_with_move(move)

    def __str__(self):
        return "training MCTS {}".format(self.player)



"""
        state = np.reshape(env.state_, (1,) + env.state_.shape)  # state.shape (1, 5, 9, 4)

        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            # return np.random.choice(self.action_dim)
            return np.random.choice(sensible_moves)
        return np.argmax(q_value)
        """



class MCTSPlayer_leaf(object):
    """Force the transition to the tree node even during self-play."""
    """AI player based on MCTS"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp, return_prob=0):  # env.state_.shape = (5,9,4)
        available = np.where(env.state_[3].flatten() == 0)[0]
        # available = [i for i in range(36) if env.state_[3][i // 4][i % 4] != 1]
        sensible_moves = available
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)  # board.shape = (5,9,4)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(-1)
            else:
                move = np.random.choice(acts, p=probs)
                # reset the root node
                assert len(np.where(np.abs(env.state_[3].reshape((-1,))-1 ))[0]) == len(self.mcts._root.children)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def oppo_node_update(self, move):
        self.mcts.update_with_move(move)

    def __str__(self):
        return "forcing leaf node MCTS {}".format(self.player)


class RandomAction(object):

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        available = [i for i in range(36) if env.state_[3][i // 4][i % 4] != 1]
        sensible_moves = available

        if len(sensible_moves) > 0:
            move = random.choice(sensible_moves)
            return move
        else:
            print("WARNING: the board is full")