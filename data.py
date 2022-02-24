import json

import pandas as pd


def save_metadata(path, **metadata):
    """Save keyword arguments as metadata in a JSON file.

    Args:
        path (str): The path to the directory in which all files will be saved.
        **metadata: A dictionary of metadata to save.
    """
    with open(f'{path}/metadata.json', 'w') as f:
        json.dump(metadata, f)


def load_metadata(path):
    """Load metadata from a directory.

    Returns:
        Dict: A dictionary of metadata.
    """
    with open(f'{path}/metadata.json', 'w') as f:
        metadata = json.load(f)
    return metadata


def save_data(path, name, game, returns_log, action_probs_log, com_probs_log, state_dist_log, runs, episodes):
    """
    This function will save all of the results to disk in CSV format for later analysis.
    :param path: The path to the directory in which all files will be saved.
    :param name: The name of the experiment.
    :param game: The name of the game that was played.
    :param returns_log: The log for the returns.
    :param action_probs_log: The log for the action probabilities.
    :param action_probs_log: The log for the communication probabilities.
    :param state_dist_log: The state distribution log in the last 10% of episodes.
    :param runs: The number of trials that were ran.
    :param episodes: The number of episodes in each run.
    :return: /
    """
    print("Saving data to disk")
    num_agents = len(returns_log)  # Extract the number of agents that were in the experiment.
    num_actions = len(action_probs_log[0][0]) - 2  # Extract the number of actions that were possible in the experiment.
    returns_columns = ['Trial', 'Episode', 'Payoff']
    action_columns = [f'Action {a + 1}' for a in range(num_actions)]
    action_columns = ['Trial', 'Episode'] + action_columns
    com_columns = ['Trial', 'Episode', 'Communication', 'No communication']

    for idx in range(num_agents):
        df_r = pd.DataFrame(returns_log[idx], columns=returns_columns)
        df_a = pd.DataFrame(action_probs_log[idx], columns=action_columns)
        df_r.to_csv(f'{path}/{name}_{game}_A{idx + 1}_returns.csv', index=False)
        df_a.to_csv(f'{path}/{name}_{game}_A{idx + 1}_probs.csv', index=False)

    if name in ['opt_comp_action', 'opt_coop_action', 'opt_coop_policy']:
        for idx in range(num_agents):
            df = pd.DataFrame(com_probs_log[idx], columns=com_columns)
            df.to_csv(f'{path}/{name}_{game}_A{idx + 1}_com.csv', index=False)

    state_dist_log /= runs * (0.1 * episodes)
    df = pd.DataFrame(state_dist_log)
    df.to_csv(f'{path}/{name}_{game}_states.csv', index=False, header=False)
    print("Finished saving data to disk")
