def get_move_probs(self, env, temp):  # state.shape = (5,9,4)

    """Run all playouts sequentially and return the available actions and
    their corresponding probabilities.
    state: the current game state
    temp: temperature parameter in (0, 1] controls the level of exploration
    """
    for n in range(self._n_playout):
        self._playout(copy.deepcopy(env))  # state_copy.shape = (5,9,4)

    # calc the move probabilities based on visit counts at the root node
    act_visits = [(act, node._n_visits)
                  for act, node in self._root._children.items()]

    acts, visits = zip(*act_visits)
    act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

    return acts, act_probs