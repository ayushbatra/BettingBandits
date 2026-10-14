import matplotlib.pyplot as plt

from learning_algorithms import UCB, NPTS, klUCB, ImedKl, OIMED, FIMED
from bandits import *
from experiment import Experiment

print("\nRunning Experiment on a Bernoulli bandit problem\n", flush=True)
horizon = 10000
nbr_xp = 50
print(f"Horizon = {horizon}\nNumber of experiments = {nbr_xp}\n", flush=True)

########################################
#            Load Bandit               #
########################################
means = np.array([0.3, 0.4, 0.45, 0.5, 0.52, 0.55])
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

experiment = Experiment(algorithms, bandit, suffix=" figure 3")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
