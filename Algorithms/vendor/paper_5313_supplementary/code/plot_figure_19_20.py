import matplotlib.pyplot as plt

from learning_algorithms import MultinomialImed, Imed
from bandits import *
from experiment import Experiment

########################################
#             Load DSSAT               #
########################################
print("\nRunning Experiment on DSSAT\n", flush=True)
horizon = 10000
nbr_xp = 50
print(f"Horizon = {horizon}\nNumber of experiments = {nbr_xp}\n", flush=True)

########################################
#             Load DSSAT               #
########################################
print("Loading DSSAT\n", flush=True)
bandit = DssatBandit()

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    MultinomialImed(bandit),
    MultinomialImed(bandit, nbr_ticks=3),
    MultinomialImed(bandit, nbr_ticks=5),
    MultinomialImed(bandit, nbr_ticks=7),
    MultinomialImed(bandit, nbr_ticks=10),
    MultinomialImed(bandit, nbr_ticks=20),
    Imed(bandit)
]

experiment = Experiment(algorithms, bandit, suffix=" figure 19")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')

########################################
#            Load Bandit               #
########################################
means = np.array([0.05, 0.1, 0.15, 0.2, 0.22, 0.25])
bandit = BetaBandit(means)
print(f"means = {means}\n", flush=True)

########################################
#             Experiment               #
########################################
print("Launching the experiment\n", flush=True)
algorithms = [
    MultinomialImed(bandit),
    MultinomialImed(bandit, nbr_ticks=3),
    MultinomialImed(bandit, nbr_ticks=5),
    MultinomialImed(bandit, nbr_ticks=7),
    MultinomialImed(bandit, nbr_ticks=10),
    MultinomialImed(bandit, nbr_ticks=20),
    Imed(bandit)
]


experiment = Experiment(algorithms, bandit, suffix=" figure 20")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
