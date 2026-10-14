import matplotlib.pyplot as plt

from learning_algorithms import UCB, KLUCB, NPTS, Imed, ImedKl, FIMED, OIMED
from bandits import *
from experiment import Experiment

horizon = 10000
nbr_xp = 50
print(f"Horizon = {horizon}\nNumber of experiments = {nbr_xp}\n", flush=True)

########################################
#            Load Bandit               #
########################################
means = np.array([0.3, 0.4, 0.45, 0.5, 0.52, 0.55])
bandit = BetaBandit(means)
print(f"means = {means}\n", flush=True)

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    UCB(bandit),
    KLUCB(bandit),
    NPTS(bandit),
    Imed(bandit),
    ImedKl(bandit),
    FIMED(bandit),
    OIMED(bandit)
]

experiment = Experiment(algorithms, bandit, suffix=" figure 10")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')


########################################
#            Load Bandit               #
########################################
means = np.array([0.3, 0.4, 0.45, 0.5, 0.52, 0.55])
bandit = BetaBandit(means, size=50)
print(f"means = {means}\n", flush=True)

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    UCB(bandit),
    KLUCB(bandit),
    NPTS(bandit),
    Imed(bandit),
    ImedKl(bandit),
    FIMED(bandit),
    OIMED(bandit)
]

experiment = Experiment(algorithms, bandit, suffix=" figure 12")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
