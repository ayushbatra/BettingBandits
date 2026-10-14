from .utils import *
from time import time


########################################
#        Online Learning Class         #
########################################
class OnlineLearningAlgorithm:
    def __init__(self, bandit, name="Sequential", initial_index=None):
        self.bandit = bandit
        self.name = name
        self.N = self.bandit.nbr_arms
        if initial_index is None:
            self.initial_index = - np.inf
        else:
            self.initial_index = initial_index

        self.last_played_arm = None
        self.last_reward = None

        self.time = None
        self.means = None
        self.nbr_pulls = None
        self.indices = None
        self.all_selected = None
        self.regret = None
        self.run_time = None

        self.reset()

    def reset(self):
        self.time = 0
        self.means = np.zeros(self.bandit.nbr_arms, dtype=float)
        self.nbr_pulls = np.zeros(self.bandit.nbr_arms, dtype=float)
        self.indices = np.zeros(self.bandit.nbr_arms, dtype=float) + self.initial_index
        self.all_selected = False
        self.regret = []
        self.run_time = []

    def specific_reset(self):
        None

    def __repr__(self):
        res = f"{self.name} algorithm - time step = {self.time}\n"
        for i in range(self.bandit.nbr_arms):
            res += "  "
            res += str(self.bandit.arms[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res

    def __str__(self):
        res = f"{self.name} algorithm - time step = {self.time}\n"
        for i in range(self.bandit.nbr_arms):
            res += "  "
            res += str(self.bandit.arms[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res

    def compute_indices(self):
        if self.all_selected:
            self.indices = np.random.rand(self.bandit.nbr_arms)
        else:
            self.indices = self.nbr_pulls.astype(float)

    def specific_update(self, arm, r):
        None

    def update_statistics(self, arm, r):
        self.specific_update(arm, r)
        self.time += 1
        n = self.nbr_pulls[arm]
        self.means[arm] = (self.means[arm] * n + r) / (n + 1)
        self.nbr_pulls[arm] += 1

        if not self.all_selected:
            self.all_selected = np.all(self.nbr_pulls > 0)

        self.last_played_arm = arm
        self.last_reward = r

        self.compute_indices()

    def choose_an_arm(self):
        return randamin(self.indices)

    def play(self):
        arm = self.choose_an_arm()
        r = self.bandit.pull(arm)
        self.update_statistics(arm, r)
        return arm, r

    def fit(self, horizon, reset=True):
        if reset:
            self.reset()
            self.specific_reset()
        for i in range(horizon):
            self.run_time.append(time())
            arm, r = self.play()
            self.regret.append(self.bandit.regrets[arm])
        self.run_time = np.array(self.run_time)
        self.run_time = self.run_time - self.run_time[0]
        return [np.cumsum(self.regret), self.run_time]


########################################
#                 UCB                  #
########################################
class UCB(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="UCB", initial_index=+np.inf, sigma=1):
        OnlineLearningAlgorithm.__init__(self, bandit, name, initial_index)
        self.sigma = sigma

    def compute_indices(self):
        if self.all_selected:
            bonus = np.sqrt(2 * np.log(self.time) / (self.sigma * self.nbr_pulls))
            self.indices = self.means + bonus
        else:
            self.indices = - self.nbr_pulls.astype(float)

    def choose_an_arm(self):
        return randamax(self.indices)


########################################
#               kl-UCB                 #
########################################
class klUCB(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="kl-UCB", initial_index=+np.inf):
        OnlineLearningAlgorithm.__init__(self, bandit, name, initial_index)

    def compute_indices(self):
        if self.all_selected:
            for arm in range(self.bandit.nbr_arms):
                m = self.means[arm]
                n = self.nbr_pulls[arm]
                if np.isclose(m, 1.):
                    self.indices[arm] = 1.
                else:
                    self.indices[arm] = kl_ucb_bern(m, n, self.time)
        else:
            self.indices = - self.nbr_pulls.astype(float)

    def choose_an_arm(self):
        return randamax(self.indices)


########################################
#               KL-UCB                 #
########################################
class KLUCB(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="KL-UCB", initial_index=+np.inf):
        OnlineLearningAlgorithm.__init__(self, bandit, name, initial_index)
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.reset()
        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        if self.all_selected:
            for arm in range(self.bandit.nbr_arms):
                m = self.means[arm]
                samples = self.samples[arm]
                n = self.nbr_pulls[arm]
                if np.isclose(m, 1.):
                    self.indices[arm] = 1.
                else:
                    self.indices[arm] = kl_ucb(samples, m, n, self.time)
        else:
            self.indices = - self.nbr_pulls.astype(float)

    def choose_an_arm(self):
        return randamax(self.indices)


########################################
#                NPTS                  #
########################################
class NPTS(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="NPTS", upper_bound=1., rng=None):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.upper_bound = upper_bound

        self.samples = {arm: [upper_bound] for arm in range(self.bandit.nbr_arms)}
        self.nbr_pulls = np.zeros(self.bandit.nbr_arms, dtype=int) + 1
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [self.upper_bound] for arm in range(self.bandit.nbr_arms)}
        self.nbr_pulls = np.zeros(self.bandit.nbr_arms, dtype=int) + 1

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        for arm in range(self.bandit.nbr_arms):
            weights = self.rng.dirichlet([1 for _ in range(self.nbr_pulls[arm])])
            samples = np.array(self.samples[arm])
            self.indices[arm] = weights @ samples

    def choose_an_arm(self):
        arm = randamax(self.indices)
        return arm


########################################
#                IMED                  #
########################################
class Imed(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="IMED", upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.upper_bound = upper_bound
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.reset()
        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            for arm in range(self.bandit.nbr_arms):
                m = self.means[arm]
                n = self.nbr_pulls[arm]
                if np.isclose(m, max_mean):
                    self.indices[arm] = np.log(n)
                else:
                    samples = self.samples[arm]
                    self.indices[arm] = - n * kinf(samples, max_mean, self.upper_bound).fun + np.log(n)
        else:
            self.indices = self.nbr_pulls.astype(float)

