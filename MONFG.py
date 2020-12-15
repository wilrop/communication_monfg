import time
import argparse
import pandas as pd
from utils import *
from games import *
from ActorCriticSER import ActorCriticSER
from ActorCriticESR import ActorCriticESR


def get_message(agents, episode):
    """
    This function gets the message from the communicating agent.
    :param agents: The list of agents.
    :param episode: The current episode.
    :return: The selected message.
    """
    communicator = episode % len(agents)  # Select the communicator in round-robin fashion.
    message = agents[communicator].communicate()
    return communicator, message


def select_actions(agents, message):
    """
    This function selects an action from each agent's policy.
    :param agents: The list of agents.
    :param message: The message selected by the communicating agent.
    :return: A list of selected actions.
    """
    selected = []
    for agent in agents:
        selected.append(agent.select_action(message))
    return selected


def calc_payoffs(agents, actions, payoff_matrix):
    """
    This function will calculate the payoffs of the agents.
    :param agents: The list of agents.
    :param actions: The action that each agent chose.
    :param payoff_matrix: The payoff matrix.
    :return: A list of received payoffs.
    """
    payoffs = []
    for agent in agents:
        payoffs.append(payoff_matrix[actions[0]][actions[1]])  # Append the payoffs from the actions.
    return payoffs


def calc_returns(action_probs, criterion, payoff_matrix):
    """
    This function will calculate the expected returns under the given criterion.
    :param action_probs: The current action probabilities of the agents.
    :param criterion: The multi-objective criterion. Either SER or ESR.
    :param payoff_matrix: The payoff matrix.
    :return: A list of expected returns.
    """
    msg_policy = action_probs[0]
    policy_nm1 = action_probs[1][0]
    policy_m1 = action_probs[1][1]
    policy_nm2 = action_probs[2][0]
    policy_m2 = action_probs[2][1]

    if criterion == 'SER':
        expected_returns_nm = msg_policy[0] * (policy_nm2 @ (policy_nm1 @ payoff_matrix))
        expected_returns_m = msg_policy[1] * (policy_m2 @ (policy_m1 @ payoff_matrix))
        expected_returns = expected_returns_nm + expected_returns_m
        ser1 = u1(expected_returns)  # Scalarise the expected returns.
        ser2 = u2(expected_returns)

        return [ser1, ser2]
    else:
        scalarised_returns1 = scalarise_matrix(payoff_matrix, u1)  # Scalarise the possible returns.
        scalarised_returns2 = scalarise_matrix(payoff_matrix, u2)
        # esr1 = policy2 @ (policy1 @ scalarised_returns1)  # Take the expected value over them.
        # esr2 = policy2 @ (policy1 @ scalarised_returns2)
        esr1, esr2 = 0, 0
        return [esr1, esr2]


def get_action_probs(agents, communicator):
    """
    This function gets the current action probabilities from each agent.
    :param agents: A list of agents.
    :return: A list of their action probabilities.
    """
    msg_probs = agents[communicator].policy_msg
    action_probs = []
    full_probs = [msg_probs]
    for agent in agents:
        policy_m = agent.policy_m
        policy_nm = agent.policy_nm
        probs = policy_nm * msg_probs[0] + policy_m * msg_probs[1]
        action_probs.append(probs)
        full_probs.append([policy_nm, policy_m])
    return action_probs, full_probs


def get_comms_probs(agents):
    message_probs = []
    for agent in agents:
        message_probs.append(agent.policy_msg)
    return message_probs


def decay_params(agents, alpha_decay):
    """
    This function decays the parameters of the Q-learning algorithm used in each agent.
    :param agents: A list of agents.
    :param alpha_decay: The factor by which to decay alpha.
    :return: /
    """
    for agent in agents:
        agent.alpha_msg *= alpha_decay
        agent.alpha_q *= alpha_decay
        agent.alpha_theta *= alpha_decay


def update(agents, message, actions, payoffs):
    """
    This function gets called after every episode to update the policy of every agent.
    :param agents: A list of agents.
    :param actions: A list of each action that was chosen, indexed by agent.
    :param payoffs: A list of each payoff that was received, indexed by agent.
    :return:
    """
    for idx, agent in enumerate(agents):
        agent.update(message, actions, payoffs[idx])


def reset(num_agents, num_actions, num_objectives, alpha_msg, alpha_q, alpha_theta, opt=False):
    """
    Ths function will create fresh agents that can be used in a new trial.
    :param num_agents: The number of agents to create.
    :param num_actions: The number of actions each agent can take.
    :param num_objectives: The number of objectives they have.
    :param alpha_q: The learning rate for the Q-values.
    :param alpha_theta: The learning rate for the parameters in the policy.
    :param opt: A boolean that decides on optimistic initialization of the Q-tables.
    :return:
    """
    agents = []
    for ag in range(num_agents):
        u, du = get_u_and_du(ag + 1)  # The utility function and derivative of the utility function for this agent.
        if criterion == 'SER':
            new_agent = ActorCriticSER(ag, u, du, alpha_msg, alpha_q, alpha_theta, num_actions, num_objectives, opt)
        else:
            new_agent = ActorCriticESR(ag, u, du, alpha_msg, alpha_q, alpha_theta, num_actions, num_objectives, opt)
        agents.append(new_agent)
    return agents


def update_comms_log(agents, message, episode, comms_log):
    communicator = episode % len(agents)
    if message is None:
        comms_log[communicator][0] += 1
    else:
        comms_log[communicator][1] += 1
    return comms_log


def run_experiment(runs, episodes, criterion, payoff_matrix, opt_init):
    """
    This function will run the requested experiment.
    :param runs: The number of different runs.
    :param episodes: The number of episodes in each run.
    :param criterion: The multi-objective optimisation criterion to use.
    :param payoff_matrix: The payoff matrix for the game.
    :param opt_init: A boolean that decides on optimistic initialization of the Q-tables.
    :return: A log of payoffs, a log for action probabilities for both agents and a log of the state distribution.
    """
    # Setting hyperparameters.
    num_agents = 2
    num_actions = payoff_matrix.shape[0]
    num_objectives = 2
    alpha_q = 0.05
    alpha_msg = 0.01
    alpha_theta = 0.05
    alpha_decay = 1

    # Setting up lists containing the results.
    payoffs_log1 = []
    payoffs_log2 = []
    act_hist_log = [[], []]
    state_dist_log = np.zeros((num_actions, num_actions))
    comms_hist_log = [[], []]

    start = time.time()

    for run in range(runs):
        print("Starting run: ", run)
        agents = reset(num_agents, num_actions, num_objectives, alpha_msg, alpha_q, alpha_theta, opt_init)

        for episode in range(episodes):
            # Run one episode.
            communicator, message = get_message(agents, episode)
            actions = select_actions(agents, message)
            payoffs = calc_payoffs(agents, actions, payoff_matrix)
            update(agents, message, actions, payoffs)  # Update the current strategy based on the returns.
            decay_params(agents, alpha_decay)  # Decay the parameters after the episode is finished.

            # Get the necessary results from this episode.
            action_probs, full_probs = get_action_probs(agents, communicator)  # Get the current action probabilities of the agents.
            comms_probs = get_comms_probs(agents)
            returns = calc_returns(full_probs, criterion, payoff_matrix)  # Calculate the SER/ESR of the current strategies.

            # Append the returns under the criterion and the action probabilities to the logs.
            returns1, returns2 = returns
            a_probs1, a_probs2 = action_probs
            comms_probs1, comms_probs2 = comms_probs
            payoffs_log1.append([episode, run, returns1])
            payoffs_log2.append([episode, run, returns2])

            comms_hist_log[0].append([episode, run, comms_probs1[0], comms_probs1[1]])
            comms_hist_log[1].append([episode, run, comms_probs2[0], comms_probs2[1]])

            if num_actions == 2:
                act_hist_log[0].append([episode, run, a_probs1[0], a_probs1[1], 0])
                act_hist_log[1].append([episode, run, a_probs2[0], a_probs2[1], 0])
            elif num_actions == 3:
                act_hist_log[0].append([episode, run, a_probs1[0], a_probs1[1], a_probs1[2]])
                act_hist_log[1].append([episode, run, a_probs2[0], a_probs2[1], a_probs2[2]])
            else:
                raise Exception("This number of actions is not yet supported")

            # If we are in the last 10% of episodes we build up a state distribution log.
            if episode >= 0.9 * episodes:
                state_dist_log[actions[0], actions[1]] += 1

    end = time.time()
    elapsed_mins = (end - start) / 60.0
    print("Minutes elapsed: " + str(elapsed_mins))

    return payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, comms_hist_log


def save_data(path, name, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, coms_hist_log, runs, episodes):
    """
    This function will save all of the results to disk in CSV format for later analysis.
    :param path: The path to the directory in which all files will be saved.
    :param name: The name of the experiment.
    :param payoffs_log1: The payoff logs for agent 1.
    :param payoffs_log2: The payoff logs for agent 2.
    :param act_hist_log: The action logs for both agents.
    :param state_dist_log: The state distribution log in the last 10% of episodes.
    :param runs: The number of trials that were ran.
    :param episodes: The number of episodes in each run.
    :return: /
    """
    print("Saving data to disk")
    columns = ['Episode', 'Trial', 'Payoff']
    df1 = pd.DataFrame(payoffs_log1, columns=columns)
    df2 = pd.DataFrame(payoffs_log2, columns=columns)

    df1.to_csv(f'{path}/agent1_{name}.csv', index=False)
    df2.to_csv(f'{path}/agent2_{name}.csv', index=False)

    columns = ['Episode', 'Trial', 'Action 1', 'Action 2', 'Action 3']
    df1 = pd.DataFrame(act_hist_log[0], columns=columns)
    df2 = pd.DataFrame(act_hist_log[1], columns=columns)

    df1.to_csv(f'{path}/agent1_probs_{name}.csv', index=False)
    df2.to_csv(f'{path}/agent2_probs_{name}.csv', index=False)

    state_dist_log /= runs * (0.1 * episodes)
    df = pd.DataFrame(state_dist_log)
    df.to_csv(f'{path}/states_{name}.csv', index=False, header=False)

    columns = ['Episode', 'Trial', 'No message', 'Message']
    df1 = pd.DataFrame(comms_hist_log[0], columns=columns)
    df2 = pd.DataFrame(comms_hist_log[1], columns=columns)

    df1.to_csv(f'{path}/agent1_comms_{name}.csv', index=False)
    df2.to_csv(f'{path}/agent2_comms_{name}.csv', index=False)

    print("Finished saving data to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-game', type=str, default='game1', choices=['game1', 'game2', 'game3', 'game4', 'game5'],
                        help="which MONFG game to play")
    parser.add_argument('-criterion', type=str, default='SER', choices=['SER', 'ESR'],
                        help="optimization criterion to use")

    parser.add_argument('-name', type=str, default='opt_comms', help='The name under which to save the results')
    parser.add_argument('-runs', type=int, default=100, help="number of trials")
    parser.add_argument('-episodes', type=int, default=5000, help="number of episodes")

    # Optimistic initialization can encourage exploration.
    parser.add_argument('-opt_init', action='store_true', help="optimistic initialization")

    args = parser.parse_args()

    # Extracting the arguments.
    game = args.game
    criterion = args.criterion
    name = args.name
    runs = args.runs
    episodes = args.episodes
    opt_init = args.opt_init

    # Starting the experiments.
    payoff_matrix = get_payoff_matrix(game)
    data = run_experiment(runs, episodes, criterion, payoff_matrix, opt_init)
    payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, comms_hist_log = data

    # Writing the data to disk.
    path = create_game_path('data', criterion, game, opt_init, False)
    mkdir_p(path)
    save_data(path, name, payoffs_log1, payoffs_log2, act_hist_log, state_dist_log, comms_hist_log, runs, episodes)
