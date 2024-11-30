import numpy as np
import copy
import wandb

from fiar_env import Fiar


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def interpolate_quantiles(interpolate_pre, interpolate_aft):
    quantiles_old = np.linspace(0, 1, interpolate_pre + 1)
    quantiles_new = np.linspace(0, 1, interpolate_aft + 1)
    new_quantiles = np.interp(quantiles_new[1:-1], quantiles_old[1:-1], interpolate_pre)

    return new_quantiles


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
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000, quantiles=None,
                 epsilon=None, search_resource=None, epsilon_decay=None,rl_model=None):
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
        self.rl_model = rl_model
        self.env = Fiar()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.search_resource = search_resource
        self.quantiles = quantiles

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        depth_fre = 0
        width_fre = 0

        while (1):
            if node.is_leaf():
                if self.rl_model in ["AC", "QAC", "DQN"]:
                    depth_fre += 1
                else:
                    depth_fre += self.quantiles
                break
            assert len(np.where(np.abs(env.state_[3].reshape((-1,)) - 1))[0]) == len(node._children)

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, info = env.step(action)

        if self.rl_model in "DQN":
            available, action_probs, leaf_value = self._policy(env)
            if len(available) > 0:
                # Action probabilities need to be created as a one-hot vector
                action_probs = np.zeros_like(action_probs)

                # argmax only for the sensible moves
                leaf_value_ = leaf_value.cpu().numpy()
                masked_leaf_value = np.zeros_like(leaf_value_)
                masked_leaf_value[available] = leaf_value_[available]

                idx_max = available[np.argmax(masked_leaf_value[available])]
                action_probs[idx_max] = 1

                # add epsilon to sensible moves and discount as much as it from the action_probs[idx_max]
                action_probs[available] += self.epsilon / len(available)
                action_probs[idx_max] -= self.epsilon

                """ use oracle """
                """ calculate next node.select's node._Q """
                # leaf_temp = node._Q + (leaf_value - node._Q)/ (node._n_visits+1)
                # # do we need to calculate next node._u?
                # leaf_value = leaf_value_[leaf_temp.argmax()]

                """ use bellman expection to cal state value """
                # leaf_value = (leaf_value_*action_probs).mean() # state value 를 구하는 방식

                """ use bellman optimality to cal state value """
                leaf_value = leaf_value_[available].max()  # state value 를 구하는 방식
            else:
                # Even if len(available) == 0, there is no issue because the leaf value will eventually be set to 0.
                leaf_value = leaf_value.max()
            action_probs = zip(available, action_probs[available])

        elif self.rl_model == "QRDQN":
            available, action_probs, leaf_value = self._policy(env)

            if len(available) > 0:
                # Action probabilities need to be created as a one-hot vector
                action_probs = np.zeros_like(action_probs)
                leaf_value_ = leaf_value.cpu().mean(axis=1).flatten()

                # In the case of QRDQN, the shape of leaf_value is (batch, n_quantiles, n_actions),
                # so the average needs to be taken over the quantiles.
                idx_max = available[np.argmax(leaf_value_[available])]
                action_probs[idx_max] = 1

                # add epsilon to sensible moves and discount as much as it from the action_probs[idx_max]
                action_probs[available] += self.epsilon / len(available)
                action_probs[idx_max] -= self.epsilon

                """ use bellman optimality to cal state value """
                leaf_value = leaf_value_[available].max()
                action_probs = zip(available, action_probs[available])
            else:
                # Even if len(available) == 0, there is no issue because the leaf value will eventually be set to 0.
                leaf_value = leaf_value.cpu().mean(axis=1).max()
                action_probs = zip(available, action_probs[available])

        elif self.rl_model in ["QAC", "QRQAC"]:
            available, action_probs, leaf_value = self._policy(env)
            action_probs = zip(available, action_probs[available])
            leaf_value = leaf_value.mean()

        else:  # state version AC, QRAC
            available, action_probs, leaf_value = self._policy(env)
            action_probs = zip(available, action_probs[available])

        if self.rl_model in ["AC"]:
            width_fre += 0

        elif self.rl_model in ["QRAC"]:
            width_fre += self.quantiles

        elif self.rl_model in ["DQN", "QAC"]:
            width_fre += len(available)

        elif self.rl_model in ["QRDQN", "QRQAC"]:
            width_fre += self.quantiles * len(available)

        # Check for end of game
        end, winners = env.winner()

        if not end:
            node.expand(action_probs)
        else:
            if winners == -1:  # tie
                leaf_value = 0.0
            elif winners == env.turn():
                leaf_value = 1.0
            else:
                leaf_value = -1.0
        node.update_recursive(-leaf_value)

        return depth_fre, width_fre

    def get_move_probs(self, env, temp, return_prob=None):  # state.shape = (5,9,4)
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        depth_ = 0
        width_ = 0
        width_fre = 0
        depth_fre = 0
        search_resource = 2000

        for n in range(self._n_playout):  # for 400 times
            env_copy = copy.deepcopy(env)
            depth_fre, width_fre = self._playout(env_copy)
            depth_ += depth_fre
            width_ += width_fre
            # if self.rl_model in ["DQN", "QRDQN"]:
            #     self.update_epsilon()

        if return_prob == 1:  # if selfplay
            wandb.log({
                "selfplay/depth": depth_fre,
                "selfplay/depth_resource_usage": depth_,
                "selfplay/depth_ratio": depth_ / (depth_ + width_),

                "selfplay/width": width_fre,
                "selfplay/width_resource_usage": width_,
                "selfplay/width_ratio": width_ / (depth_ + width_),

                "selfplay/total_planning_depth": n+1,
                # "playout/remaining_resource": (search_resource - depth_ - width_) / search_resource
            })
            if self.rl_model in ["AC"]:
                wandb.log({
                    "selfplay/total_resource_usage": depth_,
                })
            else:
                wandb.log({
                    "selfplay/total_resource_usage": depth_ + width_,
                })
        else:  # eval
            wandb.log({
                "eval/depth": depth_fre,
                "eval/depth_resource_usage": depth_,
                "eval/depth_ratio": depth_ / (depth_ + width_),

                "eval/width": width_fre,
                "eval/width_resource_usage": width_,
                "eval/width_ratio": width_ / (depth_ + width_),

                "eval/total_planning_depth": n + 1
            })

            if self.rl_model in ["AC"]:
                wandb.log({
                    "eval/total_resource_usage": depth_,
                })
            else:
                wandb.log({
                    "eval/total_resource_usage": depth_ + width_,
                })

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

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000,
                 quantiles=None, epsilon=None,  epsilon_decay=None, min_epsilon=None,
                 is_selfplay=0, elo=None, rl_model=None):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, quantiles,
                         epsilon, epsilon_decay, min_epsilon, rl_model=rl_model)

        self._is_selfplay = is_selfplay
        init_elo = 1500
        self.elo = elo if elo is not None else init_elo
        self.rl_model = rl_model

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=1e-3, return_prob=0):  # env.state_.shape = (5,9,4)
        sensible_moves = np.nonzero(env.state_[3].flatten() == 0)[0]
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp, return_prob)  # env.state_.shape = (5,9,4)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
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

    def __str__(self):
        return "training MCTS {}".format(self.player)