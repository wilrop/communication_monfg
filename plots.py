import argparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)


def save_plot(plot_name, filetype, dpi=300):
    """
    This function saves a created plot with the requested name and filetype.
    :param plot_name: The file to save the plot to.
    :param filetype: The filetype for the figure.
    :param dpi: The resolution for the final figure. Used when saving as png.
    :return: /
    """
    if filetype == 'png':
        plt.savefig(plot_name + ".png", dpi=dpi)
    else:
        plt.savefig(plot_name + ".pdf")

    plt.clf()


def plot_returns(path_plots, filetype, game, name, ag1_data, ag2_data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """
    This function will plot the returns obtained by both agents in one graph.
    :param path_plots: The path to which we save the plot.
    :param filetype: The filetype to save the file under.
    :param game: The game that was played.
    :param name: The name of the experiment.
    :param ag1_data: The data for agent 1.
    :param ag2_data: The data for agent 2.
    :param min_x: The minimum value for the x-axis.
    :param max_x: The maximum value for the x-axis.
    :param min_y: The minimum value for the y-axis.
    :param max_y: The maximum value for the y-axis.
    :return: /
    """
    print("Plotting scalarised expected returns")
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag1_data, ci='sd', label='Agent 1')
    ax = sns.lineplot(x='Episode', y='Payoff', linewidth=2.0, data=ag2_data, ci='sd', label='Agent 2')

    ax.set(ylabel='Scalarised Expected Returns')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_returns"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print("Finished plotting scalarised expected returns")


def plot_action_probabilities(path_plots, filetype, game, name, agent, data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """
    This function will plot the action probabilities for a given agent.
    :param path_plots: The path to which we save the plot.
    :param filetype: The filetype to save the file under.
    :param game: The game that was played.
    :param name: The name of the experiment that was ran.
    :param agent: The agent that is being plotted.
    :param data: The data for this agent.
    :param min_x: The minimum value for the x-axis.
    :param max_x: The maximum value for the x-axis.
    :param min_y: The minimum value for the y-axis.
    :param max_y: The maximum value for the y-axis.
    :return:
    """
    print("Plotting action probabilities for agent: " + repr(agent))
    if game == 'game2':
        label2 = 'R'
    else:
        label2 = 'M'

    ax = sns.lineplot(x='Episode', y='Action 1', linewidth=2.0, data=data, ci='sd', label='L')
    ax = sns.lineplot(x='Episode', y='Action 2', linewidth=2.0, data=data, ci='sd', label=label2)

    if game in ['game1', 'game5', 'game7']:
        ax = sns.lineplot(x='Episode', y='Action 3', linewidth=2.0, data=data, ci='sd', label='R')

    ax.set(ylabel='Action probability')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_{agent}_probs"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print("Finished plotting action probabilities for agent: " + repr(agent))


def plot_state_distribution(path_plots, game, name, data):
    """
    This function will plot the state distribution as a heatmap.
    :param path_plots: The path to which we save the plot.
    :param game: The game that was played.
    :param name: The name of the experiment that was ran.
    :param data: The state distribution data.
    :return: /
    """
    print("Plotting the state distribution.")
    if game in ['game1', 'game5', 'game7']:
        x_axis_labels = ["L", "M", "R"]
        y_axis_labels = ["L", "M", "R"]
    elif game in ['game3', 'game4', 'game6']:
        x_axis_labels = ["L", "M"]
        y_axis_labels = ["L", "M"]
    else:
        x_axis_labels = ["L", "R"]
        y_axis_labels = ["L", "R"]

    if game == 'game5':
        ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                         yticklabels=y_axis_labels, cbar=True)
    else:
        ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, xticklabels=x_axis_labels,
                         yticklabels=y_axis_labels, cbar=False)

    plot_name = f"{path_plots}/{name}_{game}_states"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print("Finished plotting the state distribution.")


def plot_com_probabilities(path_plots, filetype, game, name, agent, data, min_x=0, max_x=5000, min_y=0, max_y=20):
    """
    This function will plot the message probabilities for a given agent.
    :param path_plots: The path to which we save the plot.
    :param filetype: The filetype to save the file under.
    :param game: The game that was played.
    :param name: The name of the experiment that was ran.
    :param agent: The agent that is being plotted.
    :param data: The data for this agent.
    :param min_x: The minimum value for the x-axis.
    :param max_x: The maximum value for the x-axis.
    :param min_y: The minimum value for the y-axis.
    :param max_y: The maximum value for the y-axis.
    :return:
    """
    print("Plotting message probabilities for agent: " + repr(agent))

    ax = sns.lineplot(x='Episode', y='No communication', linewidth=2.0, data=data, ci='sd', label='No Communication')
    ax = sns.lineplot(x='Episode', y='Communication', linewidth=2.0, data=data, ci='sd', label='Communication')

    ax.set(ylabel='Communication probability')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{path_plots}/{name}_{game}_{agent}_com"
    plt.tight_layout()
    save_plot(plot_name, filetype)

    print("Finished plotting message probabilities for agent: " + repr(agent))


def plot_results(games, name, filetype, opt_init):
    """
    This function will call the different plotting functions that we need for each requested game.
    :param games: The games we want to plot the results for.
    :param name: The name of the experiment.
    :param filetype: The filetype to save the file under.
    :param opt_init: Whether optimistic initialization was used
    :return:
    """
    for game in games:
        print(f"Generating plots for {game} from experiment {experiment}")
        # Get all the paths in order.
        path_data = create_game_path('data', name, game, opt_init)
        path_plots = create_game_path('plots', name, game, opt_init)
        mkdir_p(path_plots)

        # Plot the returns for both actions in one plot.
        df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_returns.csv')
        df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_returns.csv')
        min_x = df1['Episode'].min()
        max_x = df1['Episode'].max() + 1
        min_y = 0
        max_y = 17.5
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]
        plot_returns(path_plots, filetype, game, name, df1, df2, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

        # Plot the action probabilities for both agents in a separate plot.
        df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_probs.csv')
        df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_probs.csv')
        min_y = 0
        max_y = 1
        df1 = df1.iloc[::5, :]
        df2 = df2.iloc[::5, :]
        plot_action_probabilities(path_plots, filetype, game, name, 'A1', df1, min_x=min_x, max_x=max_x, min_y=min_y,
                                  max_y=max_y)
        plot_action_probabilities(path_plots, filetype, game, name, 'A2', df2, min_x=min_x, max_x=max_x, min_y=min_y,
                                  max_y=max_y)

        # Plot the state distribution.
        df = pd.read_csv(f'{path_data}/{name}_{game}_states.csv', header=None)
        plot_state_distribution(path_plots, game, name, df)

        # Plot the message probabilities if applicable.
        if name in ['opt_comp_action', 'opt_coop_action', 'opt_coop_policy']:
            df1 = pd.read_csv(f'{path_data}/{name}_{game}_A1_com.csv')
            df2 = pd.read_csv(f'{path_data}/{name}_{game}_A2_com.csv')
            min_y = 0
            max_y = 1
            df1 = df1.iloc[::5, :]
            df2 = df2.iloc[::5, :]
            plot_com_probabilities(path_plots, filetype, game, name, 'A1', df1, min_x=min_x, max_x=max_x, min_y=min_y,
                                   max_y=max_y)
            plot_com_probabilities(path_plots, filetype, game, name, 'A2', df2, min_x=min_x, max_x=max_x, min_y=min_y,
                                   max_y=max_y)

        print("Finished generating plots for: " + repr(game))
        print("------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-games', type=str, default=['game1'], nargs='+', help="Which games to plot results for.")
    parser.add_argument('--experiment', type=str, default='no_com', help='The experiment that was ran.')
    parser.add_argument('-filetype', type=str, default='pdf', help="The filetype to save the plots under.")
    parser.add_argument('-opt_init', action='store_true', help="Whether optimistic initialization was used.")

    args = parser.parse_args()

    # Extracting the arguments.
    games = args.games
    experiment = args.experiment
    filetype = args.filetype
    opt_init = args.opt_init

    # Plotting the results for the requested games
    plot_results(games, experiment, filetype, opt_init)
