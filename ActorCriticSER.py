import random
import numpy as np
from scipy.optimize import minimize


class ActorCriticSER:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, utility_function, derivative, alpha_q, alpha_theta, num_actions, num_objectives, opt=False):
        self.utility_function = utility_function
        self.derivative = derivative
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.theta = np.zeros(num_actions)
        self.policy = softmax(self.theta)
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        # optimistic initialization of Q-table
        if opt:
            self.q_table = np.ones((num_actions, num_objectives)) * 20
        else:
            self.q_table = np.zeros((num_actions, num_objectives))

    def update(self, action, reward):
        self.update_q_table(self, action, reward)
        self.update_theta(reward)

    def update_q_table(self, action, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param action: The action played by the agent itself.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.q_table[action] += self.alpha_q * (reward - self.q_table[action])

    def update_theta(self, reward):
        """
        This method will update the parameters theta of the policy.
        :param reward: The obtained reward by this agent.
        :return: /
        """
        expected_q = np.dot(self.policy, self.q_table)
        # We apply the chain rule to calculate the gradient.
        grad_u = self.derivative(expected_q)  # The gradient of u
        grad_pg = np.dot(softmax_grad(self.policy).T, self.q_table)  # The gradient of the softmax function
        grad_theta = np.dot(grad_u, grad_pg.T)  # The gradient of the complete function J(theta).

        self.theta += self.alpha_theta * grad_theta
        self.policy = softmax(self.theta)

    def select_commit_strategy(self):
        """
        This method will determine what action this agent will publish.
        :return: The current learned policy.
        """
        return self.policy

    def select_counter_action(self, state):
        """
        This method will perform epsilon greedy action selection.
        :param state: The message from an agent in the form of their preferred joint action.
        :return: The selected action.
        """
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

    def select_committed_action(self):
        """
        This method uses the committed strategy to select the action that will be played.
        :return: An action that was selected using the current policy.
        """
        return np.random.choice(range(self.num_actions), p=self.policy)
