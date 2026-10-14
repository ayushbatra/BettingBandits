import matplotlib.pyplot as plt

from learning_algorithms import OIMED
from bandits import *
from experiment import Experiment

########################################
#             Load DSSAT               #
########################################
print("\nRunning Experiment on DSSAT\n", flush=True)
horizon = 10000
nbr_xp = 1500
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
    OIMED(bandit, name=f"OIMED - lr={1}", eta=lambda n: np.sqrt(np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={100}", eta=lambda n: np.sqrt(100 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={50}", eta=lambda n: np.sqrt(50 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={10}", eta=lambda n: np.sqrt(10 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={5}", eta=lambda n: np.sqrt(5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.5}", eta=lambda n: np.sqrt(0.5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.1}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.05}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.01}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
]

experiment = Experiment(algorithms, bandit, suffix=" figure 16")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')

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
    OIMED(bandit, name=f"OIMED - lr={1}", eta=lambda n: np.sqrt(np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={100}", eta=lambda n: np.sqrt(100 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={50}", eta=lambda n: np.sqrt(50 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={10}", eta=lambda n: np.sqrt(10 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={5}", eta=lambda n: np.sqrt(5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.5}", eta=lambda n: np.sqrt(0.5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.1}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.05}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.01}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
]

experiment = Experiment(algorithms, bandit, suffix=" figure 17")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')

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
    OIMED(bandit, name=f"OIMED - lr={1}", eta=lambda n: np.sqrt(np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={100}", eta=lambda n: np.sqrt(100 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={50}", eta=lambda n: np.sqrt(50 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={10}", eta=lambda n: np.sqrt(10 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={5}", eta=lambda n: np.sqrt(5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.5}", eta=lambda n: np.sqrt(0.5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.1}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.05}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.01}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
]


experiment = Experiment(algorithms, bandit, suffix=" figure 18")
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
    OIMED(bandit, name=f"OIMED - lr={1}", eta=lambda n: np.sqrt(np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={100}", eta=lambda n: np.sqrt(100 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={50}", eta=lambda n: np.sqrt(50 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={10}", eta=lambda n: np.sqrt(10 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={5}", eta=lambda n: np.sqrt(5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.5}", eta=lambda n: np.sqrt(0.5 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.1}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.05}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
    OIMED(bandit, name=f"OIMED - lr={0.01}", eta=lambda n: np.sqrt(0.1 * np.log(2) / (4 * n))),
]


experiment = Experiment(algorithms, bandit, suffix=" figure 18 bis")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
