import random
import math


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def ucb_score(node):
        if node.visits == 0:
            return float('inf')
        exploitation = node.value / node.visits
        exploration = math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        return exploitation + exploration

    def select(node):
        return max(node.children, key=node.ucb_score)

    def expand(node):
        actions = node.state.get_possible_actions()
        action = random.choice(actions)
        next_state = node.state.perform_action(action)
        child = Node(next_state, parent=node)
        node.children.append(child)
        return child

    def simulate(state):
        while not state.is_terminal():
            actions = state.get_possible_actions()
            action = random.choice(actions)
            state = state.perform_action(action)
        return state.get_score()

    def backpropagate(node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

    def best_child(node):
        return max(node.children, key=lambda child: child.value / child.visits)
