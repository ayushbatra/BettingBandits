import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
import datetime
from time import time
import copy
import pickle
import os
import matplotlib.pyplot as plt
import itertools


########################################
#             Experiments              #
########################################
class Experiment:
    def __init__(self, sequential_algorithms, bandit, num_cores=None, suffix=""):
        assert len(sequential_algorithms) > 0
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        print(f"Experiment running on {self.num_cores} cores")

        self.algorithms = sequential_algorithms
        self.bandit = bandit
        self.nbr_algo = len(sequential_algorithms)

        # Create Dictionary for results
        self.stat_xp = {}
        for algo in sequential_algorithms:
            self.stat_xp[algo.name] = {}

        # Create folder for results
        date = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        self.folder_name = "./results/" + bandit.name + f" {bandit.means} " + date
        self.folder_name = self.folder_name + suffix

        os.mkdir(self.folder_name)
        self.fn = self.folder_name + "/XP.pkl"

    def run(self, nbr_xp=1000, horizon=10000):
        # Save info
        f = open(self.folder_name + "/bandit.pkl", "wb")
        pickle.dump({"name": self.bandit.name, "means": self.bandit.means, "nbr_xp": nbr_xp, "horizon": horizon}, f)
        f.close()

        for algo in self.algorithms:
            name = algo.name

            start_time = time()
            regret_and_time = Parallel(n_jobs=self.num_cores)(
                delayed(copy.deepcopy(algo).fit)(horizon) for _ in tqdm(range(nbr_xp),
                                                                        desc=algo.name))
            delta_time = time() - start_time
            regret_and_time = np.array(regret_and_time)
            regret = regret_and_time[:, 0, :]
            run_time = regret_and_time[:, 1, :]

            self.stat_xp[name]['name'] = name
            self.stat_xp[name]['history'] = regret
            self.stat_xp[name]['time'] = delta_time
            self.stat_xp[name]['mean'] = np.mean(regret, axis=0)
            self.stat_xp[name]['median'] = np.quantile(regret, q=0.5, axis=0)
            self.stat_xp[name]['std'] = np.std(regret, axis=0)
            self.stat_xp[name]['quantile'] = np.quantile(regret, q=[0.1, 0.9], axis=0)

            self.stat_xp[name]['run time'] = np.mean(run_time, axis=0)
            self.stat_xp[name]['run time quantile'] = np.quantile(run_time, q=[0.05, 0.95], axis=0)

        # save experiments
        f = open(self.fn, "wb")
        pickle.dump(self.stat_xp, f)
        f.close()

        return self.stat_xp

    def plot_run_time(self, dpi=96, scale_x=1080, scale_y=566):
        # Style
        fig = plt.figure(figsize=(scale_x / dpi, scale_y / dpi), dpi=dpi)
        ax = fig.add_subplot()
        for pos in ["top", "right"]:
            ax.spines[pos].set_visible(False)

        # title = "Run time as a function of the number of samples"
        y_title = "run time"
        x_title = "number of samples"

        # plt.title(title, fontsize=20, fontweight='bold', fontname="Noto Serif")
        plt.ylabel(y_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
        plt.xlabel(x_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
        marker = itertools.cycle((',', '+', 'o', '*', 'x', 's', 'v', 'P'))
        # Plot
        for algo in self.stat_xp:
            data = self.stat_xp[algo]
            values = data["run time"]
            lab = data['name'] + "\n" + f"T={values[-1]:.1f} s"

            plt.plot(values, label=lab, marker=next(marker), markevery=0.2, linewidth=2.5)
            quantile = data["run time quantile"]
            ticks = np.arange(len(values))
            plt.fill_between(ticks,
                             quantile[0],
                             quantile[1],
                             alpha=0.1)

        plt.legend(prop={'size': 12}, loc='upper left')
        fn = f"{self.folder_name}/plot_{y_title}.pdf"
        plt.savefig(fn, dpi=dpi, bbox_inches='tight')

    def plot_stat(self, stat, uncertainty="std",
                  dpi=96, scale_x=1080, scale_y=566,
                  x_scale="linear"):
        # Style
        fig = plt.figure(figsize=(scale_x / dpi, scale_y / dpi), dpi=dpi)
        ax = fig.add_subplot()
        for pos in ["top", "right"]:
            ax.spines[pos].set_visible(False)

        # title = "Regret as a function of the number of samples"
        if stat == "mean":
            y_title = "Mean regret"
        elif stat == "median":
            y_title = "Median regret"
        else:
            y_title = "Regret"
        x_title = "Number of samples"
        if x_scale != "linear":
            x_title += " (log scale)"

        # plt.title(title, fontsize=20, fontweight='bold', fontname="Noto Serif")
        plt.ylabel(y_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
        plt.xlabel(x_title, fontsize=18, fontweight='medium', fontname="Noto Serif")
        marker = itertools.cycle((',', '+', 'o', '*', 'x', 's', 'v', 'P'))
        # Plot
        for algo in self.stat_xp:
            data = self.stat_xp[algo]
            values = data[stat]
            ticks = np.arange(len(values))
            lab = data['name'] + "\n" + f"R={values[-1]:.2f}"
            if x_scale == "linear":
                plt.plot(values, label=lab, marker=next(marker), markevery=0.2, linewidth=2.5)
            else:
                plt.semilogx(values, label=lab)
            if x_scale == "linear":
                if uncertainty == "std":
                    std = data["std"]
                    plt.fill_between(ticks,
                                     np.maximum(0, values - std),
                                     values + std,
                                     alpha=0.1)
                else:
                    quantile = data["quantile"]
                    plt.fill_between(ticks,
                                     quantile[0],
                                     quantile[1],
                                     alpha=0.1)

        plt.legend(prop={'size': 12}, loc='upper left')
        fn = f"{self.folder_name}/plot_{y_title}_{x_scale}_{uncertainty}.pdf"
        plt.savefig(fn, dpi=dpi, bbox_inches='tight')

    def plot(self):
        self.plot_stat("mean")
        self.plot_stat("median", uncertainty="quantile")
        self.plot_stat("mean", uncertainty="quantile")
        self.plot_stat("mean", x_scale="log")
        self.plot_stat("median", x_scale="log")
        self.plot_run_time()
