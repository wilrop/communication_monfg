import random
import numpy as np
from utils import *
from scipy.optimize import minimize


class ActorCriticSER:
    """
    This class represents an agent that uses the SER optimisation criterion.
    """

    def __init__(self, id, utility_function, derivative, epsilon, alpha_q, alpha_theta, num_actions, num_objectives, opt=False, rand_prob=False):
        self.id = id
        self.utility_function = utility_function
        self.derivative = derivative
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.alpha_q = alpha_q
        self.alpha_theta = alpha_theta
        self.theta_m = np.zeros(num_actions)
        self.theta_nm = np.zeros(num_actions)
        self.policy_m = softmax(self.theta_m)
        self.policy_nm = softmax(self.theta_nm)
        self.policy = np.full(num_actions, 1.0 / num_actions)
        self.op_policy = np.full(num_actions, 1.0 / num_actions)
        self.num_messages = 2
        self.msg_strategy = np.full(self.num_messages, 1.0 / self.num_messages)
        # optimistic initialization of Q-table
        if opt:
            self.msg_q_table = np.ones((2, num_objectives))
            self.q_table_m = np.ones((num_actions, num_actions, num_objectives)) * 20
            self.q_table_nm = np.ones((num_actions, num_objectives)) * 20
        else:
            self.msg_q_table = np.zeros((2, num_objectives))
            self.q_table_m = np.zeros((num_actions, num_actions, num_objectives)) * 20
            self.q_table_nm = np.zeros((num_actions, num_objectives)) * 20
        self.rand_prob = rand_prob
        self.communicator = False

    def update(self, message, actions, reward):
        """
        This method updates the Q table and policy of the agent.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained in this episode.
        :return: /
        """
        self.msg_strategy = self.calc_mixed_strategy_nonlinear()
        if message is None:
            self.update_msg_q_table(0, reward)
            self.update_q_table(message, self.q_table_nm, actions, reward)
            self.policy_nm = self.update_policy(message, self.policy_nm, self.theta_nm, self.q_table_nm)
        else:
            self.update_msg_q_table(1, reward)
            self.update_q_table(message, self.q_table_m, actions, reward)
            self.policy_m = self.update_policy(message, self.policy_m, self.theta_m, self.q_table_m)
        self.policy = self.msg_strategy[0] * self.policy_nm + self.msg_strategy[1] * self.policy_m

    def update_msg_q_table(self, communicated, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        self.msg_q_table[communicated] += self.alpha_q * (reward - self.msg_q_table[communicated])

    def update_q_table(self, message, q_table, actions, reward):
        """
        This method will update the Q-table based on the message, chosen actions and the obtained reward.
        :param actions: The actions taken by the agents.
        :param reward: The reward obtained by this agent.
        :return: /
        """
        if message is None:
            q_table[actions[self.id]] += self.alpha_q * (reward - q_table[actions[self.id]])
        else:
            q_table[actions[0], actions[1]] += self.alpha_q * (reward - q_table[actions[0], actions[1]])

    def update_policy(self, msg, policy, theta, q_table):
        """
        This method will update the parameters theta of the policy.
        :return: /
        """
        if msg is None:
            expected_q = q_table
        else:
            if self.id == 0:
                expected_q = self.op_policy @ q_table
            else:
                # We have to transpose axis 0 and 1 to interpret this as the column player.
                expected_q = self.op_policy @ q_table.transpose((1, 0, 2))

        expected_u = policy @ expected_q
        # We apply the chain rule to calculate the gradient.
        grad_u = self.derivative(expected_u)  # The gradient of u
        grad_pg = softmax_grad(policy).T @ expected_q  # The gradient of the softmax function
        grad_theta = grad_u @ grad_pg.T  # The gradient of the complete function J(theta).
        theta += self.alpha_theta * grad_theta
        return softmax(theta)

    def communicate(self):
        if random.uniform(0.0, 1.0) < self.epsilon:
            options = [None, self.policy_m]
            return options[np.random.randint(len(options))]
        else:
            message = self.select_message_greedy_mixed_nonlinear()
            if message == 0:  # Don't communicate
                return None
            else:
                self.communicator = True
                return self.policy_m

    def select_message_greedy_mixed_nonlinear(self):
        """
        This method will perform greedy action selection based on nonlinear optimiser mixed strategy search.
        :return: The selected action.
        """
        return np.random.choice(range(self.num_messages), p=self.msg_strategy)

    def calc_mixed_strategy_nonlinear(self):
        """
        This method will calculate a mixed strategy based on the nonlinear optimization.
        :return: A mixed strategy.
        """
        if self.rand_prob:
            s0 = np.random.random(self.num_messages)
            s0 /= np.sum(s0)
        else:
            s0 = np.full(self.num_messages, 1.0 / self.num_messages)  # initial guess set to equal prob over all actions.

        b = (0.0, 1.0)
        bounds = (b,) * self.num_messages  # Each pair in x will have this b as min, max
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
        This method will calculate the SER from a mixed strategy.
        :param strategy: The mixed strategy.
        :return: The SER.
        """
        expected_vec = self.calc_expected_vec(strategy)
        ser = self.utility_function(expected_vec)
        return ser

    def calc_expected_vec(self, strategy):
        """
        This method calculates the expected payoff vector for a given strategy using the agent's own Q values.
        :param strategy: The mixed strategy.
        :return: The expected results for all objectives.
        """
        expected_vec_nm = self.policy_nm @ self.q_table_nm
        expected_vec_m = self.policy_m @ (self.op_policy @ self.q_table_m)
        expected_vec = strategy[0] * expected_vec_nm + strategy[1] * expected_vec_m
        return expected_vec

    def select_action(self, message):
        """
        This method will select an action based on the message that was sent.
        :param message: The message that was sent.
        :return: The selected action.
        """
        if message is None:
            return np.random.choice(range(self.num_actions), p=self.policy_nm)
        else:
            if self.communicator:
                self.communicator = False
                return self.select_committed_strategy(message)
            else:
                return self.select_counter_action(message)

    def select_counter_action(self, op_policy):
        """
        This method will perform epsilon greedy action selection.
        :param op_policy: The strategy committed to by the opponent.
        :return: The selected action.
        """
        self.op_policy = op_policy
        self.policy_m = self.update_policy(op_policy, self.policy_m, self.theta_m, self.q_table_m)
        return np.random.choice(range(self.num_actions), p=self.policy_m)

    def select_committed_strategy(self, strategy):
        """
        This method uses the committed strategy to select the action that will be played.
        :return: An action that was selected using the current policy.
        """
        return np.random.choice(range(self.num_actions), p=strategy)
