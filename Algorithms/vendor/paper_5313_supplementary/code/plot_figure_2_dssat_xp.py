import matplotlib.pyplot as plt

from learning_algorithms import UCB, NPTS, KLUCB, klUCB, Imed, ImedKl, OIMED, FIMED
from bandits import *
from experiment import Experiment

horizon = 5000
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
    UCB(bandit),
    NPTS(bandit),
    KLUCB(bandit),
    klUCB(bandit),
    Imed(bandit),
    ImedKl(bandit),
    FIMED(bandit),
    OIMED(bandit)
]

experiment = Experiment(algorithms, bandit, suffix=" figure 2")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
