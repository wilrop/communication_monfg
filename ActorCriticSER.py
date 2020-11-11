import random
import numpy as np
from utils import *
from scipy.optimize import minimize


class ActorCriticSER:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, utility_function, derivative, alpha_q, alpha_theta, num_actions, num_objectives, opt=False):
        self.utility_function = utility_function
        self.derivative = derivative
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        self.op_policy = np.full(num_actions, 1 / num_actions)
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_actions, num_objectives))

    def update(self, action1, action2, reward):
        """
        This method updates the Q table and policy of the agent.
        :param action1: The action taken by the opponent.
        :param action2: The action taken by the agent itself.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.update_q_table(action1, action2, reward)
        self.update_policy()

    def update_q_table(self, action1, action2, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param action1: The action played by the opponent.
        :param action2: The action played by the agent itself.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[action1][action2] += self.alpha_q * (reward - self.q_table[action1][action2])

    def update_policy(self):
        """
        This method will update the parameters theta of the policy.
        :return: /
        """
        expected_q = self.op_policy @ self.q_table
        expected_u = self.policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.derivative(expected_u)  # The gradient of u
        grad_pg = softmax_grad(self.policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        self.theta += self.alpha_theta * grad_theta
        self.policy = softmax(self.theta)

    def select_commit_strategy(self):
        """
        This method will determine what action this agent will publish.
        :return: The current learned policy.
        """
        return self.policy

    def select_counter_action(self, op_policy):
        """
        This method will perform epsilon greedy action selection.
        :param op_policy: The strategy committed to by the opponent.
        :return: The selected action.
        """
        self.op_policy = op_policy
        self.update_policy()
        return np.random.choice(range(self.num_actions), p=self.policy)

    def select_committed_action(self):
        """
        This method uses the committed strategy to select the action that will be played.
        :return: An action that was selected using the current policy.
        """
        return np.random.choice(range(self.num_actions), p=self.policy)
