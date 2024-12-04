import numpy as np
import copy
import torch
import wandb
from fiar_env import Fiar


def is_float(a):
    return np.floor(a) != float(a)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def get_fixed_indices(p):
    if p == 1:
        return [13, 40, 67]
    elif p == 2:
        return [4, 13, 22, 31, 40, 49, 58, 67, 76]
    elif p == 3:
        return [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79]
    elif p == 4:
        return list(range(81))
    else:
        raise ValueError("p should be between 1 and 4")


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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=1000, epsilon=None,
                 search_resource=None, epsilon_decay=None, min_epsilon=None, rl_model=None):
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
        self.min_epsilon = min_epsilon
        self.search_resource = search_resource
        self.p = 1

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        threshold = 0.1
        depth_fre = 0
        width_fre = 0
        self.depth_count = 0

        while self.search_resource > 0:
            if node.is_leaf():
                break
            assert len(np.where(np.abs(env.state_[3].reshape((-1,)) - 1))[0]) == len(node._children)

            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, info = env.step(action)

        self.p = 1
        available, action_probs, leaf_value = self._policy(env)

        # print("available 개수 ", len(available))

        while (self.p <= 4) and (self.search_resource > 0) and (len(available) > 0):

            n_indices = get_fixed_indices(self.p)
            action_probs_ = np.zeros_like(action_probs)
            leaf_value_ = leaf_value[n_indices, :].cpu().mean(axis=0).squeeze()  # leaf_value shape : 2-dim to 1-dim

            # masked_leaf_value = np.zeros_like(leaf_value_)
            # masked_leaf_value[available] = leaf_value_[available]

            if self.rl_model == "EQRDQN":
                idx_max = available[np.argmax(leaf_value_[available])]
                action_probs_[idx_max] = 1
                action_probs_[available] += self.epsilon / len(available)
                action_probs_[idx_max] -= self.epsilon
                action_probs = action_probs_

            leaf_value_srted, idx_srted = leaf_value_.sort()

            if torch.abs(leaf_value_[idx_srted[-1]] - leaf_value_[idx_srted[-2]]) > threshold:
                # TODO action value 버전이니까 leaf_value구할 때 available하지 않은 action들에 masking 처리를 했었음.
                # TODO 근데 available 한 action의 개수가 많을때는 딱히 문제가 되지 않을거 같은데
                # TODO 예를 들어 1,2 개 남았다고 했을떄 available한 action의 value가 - 가 나오게 된다면?
                # TODO 그러면 unavailable한 action의 value값이 되게 되는데
                # TODO 다시 말해 masking한 action value 가 되어버리니까 0이 되는데
                # TODO QRDQN 은 action value를 가지고 가장 높은 value에 1을 넣어주니까 이런 문제는 발생하지 않을거 같은데
                # TODO QRQAC 은 network에 나온 값을 masking처리만 하고 사용되니까 이런 문제가 발생할 수 있을 것

                action_probs = zip(available, action_probs[available])
                masked_leaf_value = leaf_value_.mean(axis=0).squeeze()

                if self.rl_model == "EQRDQN":
                    print("heloo", len(available))
                    print(masked_leaf_value.shape())
                    leaf_value = masked_leaf_value[idx_srted[-1]]  # max Q-value
                elif self.rl_model == "EQRQAC":
                    leaf_value = masked_leaf_value[available].mean()  # max Q-value

                # self.update_depth_search_resource(self.p)
                # self.update_width_search_resource(self.p, available)  # TODO 이게 머하는건지 알아보기

                # Check for end of game
                end, winners = env.winner()

                if not end:
                    node.expand(action_probs)
                else:
                    if winners == 0:  # tie
                        leaf_value = 0.0
                    elif winners == env.turn():
                        leaf_value = 1.0
                    else:
                        leaf_value = -1.0
                node.update_recursive(-leaf_value)

                depth_fre += (3 ** self.p)
                width_fre += (3 ** self.p) * len(available)
                # print("depth uses:", depth_fre)
                # print("width uses:", width_fre)
                print("P=5가 아닐떄 값",self.p)
                print("WTF")
                return depth_fre, width_fre

            else:
                self.update_depth_search_resource(self.p)
                self.update_width_search_resource(self.p, available)
                self.p += 1

            if self.search_resource <= 0 or self.p == 5: # TODO self.p -1 로 계산되어야하는 거 아닌가
                action_probs = zip(available, action_probs[available])
                leaf_value = leaf_value_[idx_srted[-1]]

                depth_fre += (3 ** self.p)
                width_fre += (3 ** self.p) * len(available)

                # Check for end of game
                end, winners = env.winner()

                if not end:
                    node.expand(action_probs)
                else:
                    if winners == 0:  # tie
                        leaf_value = 0.0
                    elif winners == env.turn():
                        leaf_value = 1.0
                    else:
                        leaf_value = -1.0
                node.update_recursive(-leaf_value)

                depth_fre += 3 ** (self.p-1)
                width_fre += 3 ** (self.p-1) * len(available)
                # print("depth uses:", depth_fre)
                # print("width uses:", width_fre)

                # print("P=5일때 값", self.p)

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

        for n in range(self._n_playout):  # for 400 times
            env_copy = copy.deepcopy(env)
            depth_fre, width_fre = self._playout(env_copy)
            depth_ += depth_fre
            width_ += width_fre

            # print("Remain resource", self.search_resource)
            if self.search_resource <= 0:
                break

        if return_prob == 1:
            wandb.log({
                "selfplay/depth": depth_fre,
                "selfplay/depth_resource_usage": depth_,
                "selfplay/depth_ratio": depth_ / (depth_ + width_),

                "selfplay/width": width_fre,
                "selfplay/width_resource_usage": width_,
                "selfplay/width_ratio": width_ / (depth_ + width_),

                "selfplay/total_resource_usage": depth_ + width_,
                "selfplay/total_planning_depth": n + 1
            })
        else:  # eval
            wandb.log({
                "eval/depth": depth_fre,
                "eval/depth_resource_usage": depth_,
                "eval/depth_ratio": depth_ / (depth_ + width_),

                "eval/width": width_fre,
                "eval/width_resource_usage": width_,
                "eval/width_ratio": width_ / (depth_ + width_),

                "eval/total_resource_usage": depth_ + width_,
                "eval/total_planning_depth": n + 1
            })

        print("Playout times", n + 1)

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

    def update_depth_search_resource(self, p):
        if p in [1, 2, 3, 4]:
            self.search_resource -= 2 ** (p-1) * 3
        else:
            assert False, "not defined"

    def update_width_search_resource(self, p, available):
        # print("available 개수 ", len(available))
        if p in [1, 2, 3, 4]:
            self.search_resource -= 2 ** (p-1) * 3 * len(available)
        else:
            assert False, "not defined"

    def __str__(self):
        return "MCTS"


class EMCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000,
                 epsilon=None, search_resource=None, is_selfplay=0, elo=None, rl_model=None):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, epsilon,
                         search_resource, rl_model=rl_model)

        self._is_selfplay = is_selfplay
        init_elo = 1500
        self.elo = elo if elo is not None else init_elo
        self.rl_model = rl_model
        self.resource = search_resource

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=1e-3, return_prob=0):  # env.state_.shape = (5,9,4)
        sensible_moves = np.nonzero(env.state_[3].flatten() == 0)[0]
        move_probs = np.zeros(env.state_.shape[1] * env.state_.shape[2])
        self.mcts.search_resource = self.resource

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)  # env.state_.shape = (5,9,4)
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