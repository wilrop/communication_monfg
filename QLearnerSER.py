import random
import numpy as np
from scipy.optimize import minimize


class QLearnerSER:
    """
    This class represents an agent that uses the SER multi-objective optimisation criterion.
    """

    def __init__(self, id, utility, alpha, epsilon, num_actions, num_objectives, opt=False, rand_prob=False):
        self.id = id
        self.utility = utility
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_objectives))
        self.payoffs_table = np.zeros((num_actions, num_actions, num_objectives))
        self.rand_prob = rand_prob
        self.strategy = np.full(num_actions, 1 / num_actions)
        self.communicating = False
        self.latest_message = 0

    def update(self, actions, reward):
        """
        This method will update the Q-table and strategy of the agent.
        :param actions: The actions selected in the previous episode.
        :param reward: The reward that was obtained by the agent.
        :return: /
        """
        own_action = actions[self.id]
        if self.communicating:
            self.update_q_table(own_action, reward)
            self.communicating = False
        self.update_payoffs_table(actions, reward)

    def update_q_table(self, action, reward):
        """
        This method will update the Q-table based on the chosen actions and the obtained reward.
        :param action: The action chosen by this agent.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[action] += self.alpha * (reward - self.q_table[action])

    def update_payoffs_table(self, actions, reward):
        """
        This method will update the payoffs table to learn the payoff vector of joint actions.
        :param actions: The actions that were taken in the previous episode.
        :param reward: The reward obtained by this joint action.
        :return: /
        """
        self.payoffs_table[actions[0], actions[1]] += self.alpha * (reward - self.payoffs_table[actions[0], actions[1]])

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if self.communicating:
            return self.select_published_action(message)  # If this agent is committing, they must follow through.
        else:
            return self.select_counter_action(message)  # Otherwise select a counter action.

    def select_commit_action(self):
        """
        This method will determine what action this agent will publish.
        :return: The action that will maximise this agent's SER, given that the other agent also maximises its response.
        """
        self.communicating = True
        self.strategy = self.calc_mixed_strategy_nonlinear()
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy_mixed_nonlinear()

    def select_counter_action(self, state):
        """
        This method will perform epsilon greedy action selection.
        :param state: The message from an agent in the form of their preferred joint action.
        :return: The selected action.
        """
        self.latest_message = state
        self.strategy = self.calc_mixed_strategy_nonlinear()
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_action_greedy_mixed_nonlinear()

    def select_random_action(self):
        """
        This method will return a random action.
        :return: An action (an integer value).
        """
        return np.random.randint(self.num_actions)

    def select_action_greedy_mixed_nonlinear(self):
        """
        This method will perform greedy action selection based on nonlinear optimiser mixed strategy search.
        :return: The selected action.
        """
        return np.random.choice(range(self.num_actions), p=self.strategy)

    def calc_mixed_strategy_nonlinear(self):
        """
        This method will calculate a mixed strategy based on the nonlinear optimization.
        :return: A mixed strategy.
        """
        if self.rand_prob:
            s0 = np.random.random(self.num_actions)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_actions, 1.0 / self.num_actions)  # initial guess set to equal prob over all actions.

        b = (0.0, 1.0)
        bounds = (b,) * self.num_actions  # Each pair in x will have this b as min, max
        con1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        solution = minimize(self.objective, s0, bounds=bounds, constraints=con1)
        strategy = solution.x

        if np.sum(strategy) != 1:
            strategy = strategy / np.sum(strategy)
        return strategy

    def objective(self, strategy):
        """
        This method is the objective function to be minimised by the nonlinear optimiser.
        Therefore it returns the negative of SER.
        :param strategy: The mixed strategy for the agent.
        :return: The negative SER.
        """
        return - self.calc_ser_from_strategy(strategy)

    def calc_ser_from_strategy(self, strategy):
        """
        This method will select the action that will result in the highest utility for the agent.
        :return: The selected action.
        """
        expected_vec = self.calc_expected_vec(strategy)
        ser = self.utility(expected_vec)
        return ser

    def calc_expected_vec(self, strategy):
        """
        This method calculates the expected payoff vector for a given strategy using the agent's own Q values.
        :param strategy: The mixed strategy.
        :return: The expected results for all objectives.
        """
        if self.communicating:
            expected_vec = strategy @ self.q_table
        else:
            if self.id == 0:  # Row player
                expected_q = self.payoffs_table[:, self.latest_message]  # Column player sent a message.
            elif self.id == 1:  # Column player
                expected_q = self.payoffs_table[self.latest_message]  # row player sent a message.
            else:
                raise Exception("Player id does not exist")
            expected_vec = strategy @ expected_q

        return expected_vec

    @staticmethod
    def select_published_action(state):
        """
        This method simply plays the action that it already published.
        :param state: The action it published.
        :return: The action it published.
        """
        return state
