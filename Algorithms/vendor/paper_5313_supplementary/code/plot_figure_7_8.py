import matplotlib.pyplot as plt

from learning_algorithms import UCB, NPTS, klUCB, ImedKl, OIMED, FIMED
from bandits import *
from experiment import Experiment

horizon = 10000
nbr_xp = 50
print(f"Horizon = {horizon}\nNumber of experiments = {nbr_xp}\n", flush=True)

########################################
#            Load Bandit               #
########################################
means = np.array([0.4, 0.6, 0.7, 0.85, 0.9, 0.95])
bandit = BernoulliBandit(means)
print(f"means = {means}\n", flush=True)

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    UCB(bandit),
    NPTS(bandit),
    klUCB(bandit, name="KL-UCB"),
    ImedKl(bandit, name="IMED"),
    FIMED(bandit),
    OIMED(bandit)
]
experiment = Experiment(algorithms, bandit, suffix=" figure 7")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')


########################################
#            Load Bandit               #
########################################
means = np.array([0.05, 0.1, 0.15, 0.2, 0.22, 0.25])
bandit = BernoulliBandit(means)
print(f"means = {means}\n", flush=True)

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    UCB(bandit),
    NPTS(bandit),
    klUCB(bandit, name="KL-UCB"),
    ImedKl(bandit, name="IMED"),
    FIMED(bandit),
    OIMED(bandit)
]
experiment = Experiment(algorithms, bandit, suffix=" figure 8")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
