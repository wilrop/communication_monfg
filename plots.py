import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from games import *
from utils import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)

experiments = ['No-comms', 'Max-utility', 'Stackelberg', 'Mixed-stackelberg']
games = ['game1', 'game2', 'game3', 'game4', 'game5']
criterion = 'SER'
rand_prob = False
episodes = 10000


def plot_hist(data_path, plot_path):
    for game in games:
        plt.clf()
        data = []
        min1, max1, min2, max2 = get_min_max(game)
        for experiment in experiments:
            if experiment == 'No-comms':
                data1_path = f'{data_path}/{experiment}/data/{criterion}/{game}/zero_init/opt_eq/agent1_NE_no_comm.csv'
                data2_path = f'{data_path}/{experiment}/data/{criterion}/{game}/zero_init/opt_eq/agent2_NE_no_comm.csv'
            else:
                data1_path = f'{data_path}/{experiment}/data/{criterion}/{game}/zero_init/opt_eq/agent1_NE_comm.csv'
                data2_path = f'{data_path}/{experiment}/data/{criterion}/{game}/zero_init/opt_eq/agent2_NE_comm.csv'

            df1 = pd.read_csv(data1_path)
            df2 = pd.read_csv(data2_path)

            payoffs1 = df1['Payoff']
            payoffs2 = df2['Payoff']

            normalised1 = normalise(min1, max1, payoffs1)
            normalised2 = normalise(min2, max2, payoffs2)
            nmean1 = np.mean(normalised1)
            nmean2 = np.mean(normalised2)
            data.append([experiment, 'Agent 1', nmean1])
            data.append([experiment, 'Agent 2', nmean2])

        df = pd.DataFrame(data, columns=['Experiment', 'Agent', 'Normalised average utility'])
        sns.barplot(x="Experiment", y="Normalised average utility", hue="Agent", data=df)
        plot_name = f"{plot_path}/{game}/norm_avg_comp"
        plt.savefig(plot_name + ".pdf")


data_path = '/Users/willemropke/OneDrive - Vrije Universiteit Brussel/2 MA/Thesis/Results'
plot_path = '/Users/willemropke/OneDrive - Vrije Universiteit Brussel/2 MA/Thesis/Results/Comparisons'

plot_hist(data_path, plot_path)
