import matplotlib.pyplot as plt

from learning_algorithms import ImedSoftBayes, OIMED
from bandits import *
from experiment import Experiment


########################################
#             Load DSSAT               #
########################################
print("\nRunning Experiment on DSSAT\n", flush=True)
horizon = 10000
nbr_xp = 500
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
    OIMED(bandit),
    ImedSoftBayes(bandit)
]
experiment = Experiment(algorithms, bandit, suffix=" figure 4")
_ = experiment.run(nbr_xp, horizon)
experiment.plot()
plt.close('all')
