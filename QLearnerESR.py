import random
import numpy as np
from scipy.optimize import minimize


class QLearnerESR:
    """
    This class represents an agent that uses the ESR optimisation criterion.
    """

    def __init__(self, agent_id, alpha, gamma, epsilon, num_states, num_actions, num_objectives, opt=False,
                 rand_prob=False):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.payoffs_table = np.zeros((num_actions, num_actions))
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones(num_states) * 20
        else:
            self.q_table = np.zeros(num_states)
        self.current_state = -1
        self.rand_prob = rand_prob

    def update_q_table(self, prev_state, action, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param prev_state: The message.
        :param action: The chosen action by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        old_q = self.q_table[prev_state]
        utility = calc_utility(self.agent_id, reward)
        new_q = old_q + self.alpha * (utility - old_q)
        self.q_table[prev_state] = new_q

    def update_payoffs_table(self, actions, payoffs):
        """
        Log the payoff vector of a joint action in the payoffs table.
        :param actions: The joint action.
        :param payoffs: The multi-objective payoff vector.
        :return: /
        """
        coords = tuple(actions)
        utility = calc_utility(self.agent_id, payoffs)
        self.payoffs_table[coords] = utility

    def select_publish_action(self):
        """
        This method will determine what action this agent will publish.
        :return: The action that will maximise this agent's ESR, given that the other agent also maximises its response.
        """
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return np.argmax(self.q_table)

    def select_counter_action(self, state):
        """
        This method will perform epsilon greedy action selection.
        :param state: The message from an agent in the form of their preferred joint action.
        :return: The selected action.
        """
        self.current_state = state
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy()

    def select_random_action(self):
        """
        This method will return a random action.
        :return: An action (an integer value).
        """
        random_action = np.random.randint(self.num_actions)
        return random_action

    def select_action_greedy(self):
        """
        This method will select the action that will result in the highest utility for the agent.
        :return: The selected action.
        """
        if self.agent_id == 0:  # Row player
            array = self.payoffs_table[:, self.current_state]  # Select the column from which we can pick an action
        else:  # Column player
            array = self.payoffs_table[self.current_state]  # Select the row from which we can pick an action
        return np.argmax(array)

    @staticmethod
    def select_published_action(state):
        """
        This method simply plays the action that it already published.
        :param state: The action it published.
        :return: The action it published.
        """
        return state


def calc_utility(agent, vector):
    """
    This function will calculate the utility for an agent and a payoff vector.
    :param agent: The agent id.
    :param vector: The results for the objectives.
    :return: The utility.
    """
    utility = 0
    if agent == 0:
        utility = vector[0] ** 2 + vector[1] ** 2  # Utility function for agent 1
    elif agent == 1:
        utility = vector[0] * vector[1]  # Utility function for agent 2
    return utility


def softmax(q):
    soft_q = np.exp(q - np.max(q))
    return soft_q / soft_q.sum(axis=0)
