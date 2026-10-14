from utils import *
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


########################################
#               IMED kl                #
########################################
class ImedKl(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="IMED-kl", kl=kl_bernoulli):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.kl = kl

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            self.indices = self.nbr_pulls * self.kl(self.means, max_mean) + np.log(self.nbr_pulls)
        else:
            self.indices = self.nbr_pulls.astype(float)


########################################
#          Multinomial IMED            #
########################################
class MultinomialImed(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="Mult. IMED", upper_bound=1., lower_bound=0., nbr_ticks=2):
        OnlineLearningAlgorithm.__init__(self, bandit, f"{name} - {nbr_ticks} items")
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.nbr_ticks = nbr_ticks
        self.sigma = (upper_bound - lower_bound) / (nbr_ticks - 1)
        self.ticks = np.array([lower_bound + i * self.sigma for i in range(nbr_ticks)])

        self.samples = {arm: np.zeros(nbr_ticks) for arm in range(self.bandit.nbr_arms)}

        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: np.zeros(self.nbr_ticks) for arm in range(self.bandit.nbr_arms)}

    def update_statistics(self, arm, r):
        float_index = r / self.sigma
        left_index = int(float_index)
        index = left_index + np.random.binomial(1, float_index - left_index)
        r = self.lower_bound + index * self.sigma
        self.specific_update(arm, index)
        self.time += 1
        n = self.nbr_pulls[arm]
        self.means[arm] = (self.means[arm] * n + r) / (n + 1)
        self.nbr_pulls[arm] += 1

        if not self.all_selected:
            self.all_selected = np.all(self.nbr_pulls > 0)

        self.last_played_arm = arm
        self.last_reward = r
        self.compute_indices()

    def specific_update(self, arm, index):
        self.samples[arm][index] += 1

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
                    self.indices[arm] = - multinomial_kinf(samples, self.ticks, max_mean,
                                                           self.upper_bound).fun + np.log(n)
        else:
            self.indices = self.nbr_pulls.astype(float)


########################################
#                FIMED                 #
########################################
class FIMED(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="FIMED", upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.upper_bound = upper_bound

        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.kinf = {arm: None for arm in range(self.bandit.nbr_arms)}
        self.last_best_mean = {arm: 0 for arm in range(self.bandit.nbr_arms)}
        self.best_arm = 0
        self.first_time = True

        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.kinf = {arm: None for arm in range(self.bandit.nbr_arms)}
        self.last_best_mean = {arm: 0 for arm in range(self.bandit.nbr_arms)}
        self.best_arm = 0
        self.first_time = True

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            if self.first_time:
                self.best_arm = np.argmax(self.means)
                self.first_time = False
                for arm in range(self.bandit.nbr_arms):
                    m = self.means[arm]
                    n = self.nbr_pulls[arm]
                    samples = self.samples[arm]

                    res = kinf(samples, max_mean, self.upper_bound)
                    self.kinf[arm] = res
                    self.last_best_mean[arm] = max_mean

                    if np.isclose(m, max_mean):
                        self.indices[arm] = np.log(n)
                    else:
                        self.indices[arm] = - n * res.fun + np.log(n)
            else:
                if np.argmax(self.means) != self.best_arm:
                    self.best_arm = np.argmax(self.means)
                    for arm in range(self.bandit.nbr_arms):
                        m = self.means[arm]
                        n = self.nbr_pulls[arm]
                        samples = self.samples[arm]

                        res = kinf(samples, max_mean, self.upper_bound)
                        self.kinf[arm] = res
                        self.last_best_mean[arm] = max_mean

                        if np.isclose(m, max_mean):
                            self.indices[arm] = np.log(n)
                        else:
                            self.indices[arm] = - n * res.fun + np.log(n)
                else:
                    for arm in range(self.bandit.nbr_arms):
                        m = self.means[arm]
                        n = self.nbr_pulls[arm]
                        if arm == self.last_played_arm:
                            if np.isclose(m, max_mean):
                                self.indices[arm] = np.log(n)
                            else:
                                samples = self.samples[arm]
                                res = kinf(samples, max_mean, self.upper_bound)
                                self.kinf[arm] = res
                                self.last_best_mean[arm] = max_mean
                                self.indices[arm] = - n * res.fun + np.log(n)
                        else:
                            if np.isclose(m, max_mean):
                                self.indices[arm] = np.log(n)
                            else:
                                res = self.kinf[arm]
                                approx_kinf = - res.fun + (max_mean - self.last_best_mean[arm]) * res.x
                                self.indices[arm] = n * approx_kinf + np.log(n)
        else:
            self.indices = self.nbr_pulls.astype(float)
            self.best_arm = np.argmax(self.means)


########################################
#            IMED no duels             #
########################################
class ImedSoftBayes(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="IMED no duels", eta=lambda n: np.sqrt(np.log(2) / (4 * n)), gamma=0.001,
                 upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.name = name
        self.eta = eta
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.ub = upper_bound + gamma
        self.kl_ub = lambda x: np.log(1. / (self.ub - x))

        self.lbd = {arm: 0.5 for arm in range(self.bandit.nbr_arms)}
        self.kl = {arm: 0 for arm in range(self.bandit.nbr_arms)}

        self.specific_reset()

    def specific_reset(self):
        self.lbd = {arm: 0.5 for arm in range(self.bandit.nbr_arms)}
        self.kl = {arm: 0 for arm in range(self.bandit.nbr_arms)}

    def compute_indices(self):
        max_mean = np.max(self.means)
        arm = self.last_played_arm
        r = self.last_reward

        if np.isclose(self.means[arm], max_mean):
            self.indices[arm] = np.log(self.nbr_pulls[arm])
        elif np.isclose(max_mean, self.upper_bound):
            self.indices[arm] = np.inf
        else:
            lbd = self.lbd[arm]
            n = self.nbr_pulls[arm]

            eta = self.eta(n) / (1 + self.eta(n))
            eta_plus_un = self.eta(n + 1) / (1 + self.eta(n + 1))
            frac = (self.ub - r) / (self.ub - max_mean)
            lbd = lbd * (1 - eta + eta * frac / (1 - lbd + lbd * frac))
            lbd = lbd * (eta_plus_un / eta) + (1 - (eta_plus_un / eta)) * 0.5
            self.lbd[arm] = lbd

            self.kl[arm] += np.log(1 - lbd + lbd * frac)
            self.kl[arm] = np.minimum(self.kl[arm], n * self.kl_ub(max_mean))

            self.indices[arm] = self.kl[arm] + np.log(self.nbr_pulls[arm])


########################################
#                OIMED                 #
########################################
class OIMED(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="OIMED", eta=lambda n: np.sqrt(np.log(2) / (4 * n)),
                 gamma=0.001, upper_bound=1., greedy_duel=True):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.name = name
        self.eta = eta
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.ub = upper_bound + gamma
        self.kl_ub = lambda x: np.log(1. / (self.ub - x))
        self.gd = greedy_duel

        self.z = np.zeros(self.bandit.nbr_arms, dtype=int)
        self.nbr_pulls_lc = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float)
        self.lbd = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms)) + 0.5
        self.kl = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms))
        self.queue = [arm for arm in range(self.bandit.nbr_arms)]
        self.indices = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float) + self.initial_index
        self.leader = None

        self.specific_reset()

    def specific_reset(self):
        self.z = np.zeros(self.bandit.nbr_arms, dtype=int)
        self.nbr_pulls_lc = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float)
        self.lbd = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms)) + 0.5
        self.kl = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms))
        self.queue = [arm for arm in range(self.bandit.nbr_arms)]
        self.indices = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float) + self.initial_index
        self.leader = None

    def compute_indices(self):
        if self.leader is None:
            self.leader = randamax(self.nbr_pulls, self.means)
        if self.z[self.last_played_arm] == 1:
            self.z[self.last_played_arm] = 0
            self.nbr_pulls_lc[self.leader, self.last_played_arm] += 1

            leader = self.leader

            leader_mean = self.means[leader]
            arm = self.last_played_arm
            r = self.last_reward

            lbd = self.lbd[leader, arm]
            n = self.nbr_pulls_lc[leader, arm]

            eta = self.eta(n) / (1 + self.eta(n))
            eta_plus_un = self.eta(n + 1) / (1 + self.eta(n + 1))
            frac = (self.ub - r) / (self.ub - leader_mean)
            lbd = lbd * (1 - eta + eta * frac / (1 - lbd + lbd * frac))
            lbd = lbd * (eta_plus_un / eta) + (1 - (eta_plus_un / eta)) * 0.5
            self.lbd[leader, arm] = lbd

            if self.means[arm] >= leader_mean:
                self.kl[leader, arm] = 0.
            else:
                self.kl[leader, arm] += np.log(1 - lbd + lbd * frac)
                self.kl[leader, arm] = np.minimum(self.kl[leader, arm], n * self.kl_ub(leader_mean))
                self.kl[leader, arm] = np.maximum(0., self.kl[leader, arm])
            self.indices[leader, arm] = self.kl[leader, arm] + np.log(n)

    def duel(self):
        leader = randamax(self.nbr_pulls, self.means)
        self.leader = leader
        leader_mean = self.means[leader]
        n = self.nbr_pulls[leader]
        if self.gd:
            pull_threshold = 10 * np.sqrt(n)
        else:
            pull_threshold = np.inf

        leader_index = np.log(n)

        for challenger in range(self.bandit.nbr_arms):
            if challenger != leader:
                if (self.nbr_pulls_lc[leader, challenger] >= pull_threshold) or (leader_mean >= self.upper_bound) or (
                        self.means[challenger] >= self.upper_bound):
                    if self.means[challenger] >= leader_mean:
                        self.queue.append(challenger)
                else:
                    if self.indices[leader, challenger] <= leader_index:
                        self.queue.append(challenger)
                        self.z[challenger] = 1

        if len(self.queue) == 0:
            self.queue.append(leader)

    def choose_an_arm(self):
        if len(self.queue) == 0:
            self.duel()
            arm = self.queue.pop()
        else:
            arm = self.queue.pop()
        return arm


########################################
#                 MED                  #
########################################
class Med(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="MED", upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.upper_bound = upper_bound
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.rng = np.random.default_rng()
        self.reset()
        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.rng = np.random.default_rng()

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            for arm in range(self.bandit.nbr_arms):
                m = self.means[arm]
                n = self.nbr_pulls[arm]
                if np.isclose(m, max_mean):
                    self.indices[arm] = 0.
                else:
                    samples = self.samples[arm]
                    self.indices[arm] = n * kinf(samples, max_mean, self.upper_bound).fun
        else:
            self.indices = self.nbr_pulls.astype(float)

    def choose_an_arm(self):
        if self.all_selected:
            g = self.rng.gumbel(0, 1, self.bandit.nbr_arms)
            arm = randamax(g + self.indices)
        else:
            arm = randamin(self.indices)
        return arm


########################################
#               MED kl                 #
########################################
class MedKl(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="MED kl", kl=kl_bernoulli):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.rng = np.random.default_rng()
        self.kl = kl
        self.reset()
        self.specific_reset()

    def specific_reset(self):
        self.rng = np.random.default_rng()

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            self.indices = - self.nbr_pulls * self.kl(self.means, max_mean)
        else:
            self.indices = self.nbr_pulls.astype(float)

    def choose_an_arm(self):
        if self.all_selected:
            g = self.rng.gumbel(0, 1, self.bandit.nbr_arms)
            arm = randamax(g + self.indices)
        else:
            arm = randamin(self.indices)
        return arm


########################################
#                 FMED                 #
########################################
class FMED(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="FMED", upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.upper_bound = upper_bound
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.kinf = {arm: None for arm in range(self.bandit.nbr_arms)}
        self.last_best_mean = {arm: 0 for arm in range(self.bandit.nbr_arms)}
        self.best_arm = 0
        self.first_time = True
        self.rng = np.random.default_rng()
        self.reset()
        self.specific_reset()

    def specific_reset(self):
        self.samples = {arm: [] for arm in range(self.bandit.nbr_arms)}
        self.kinf = {arm: None for arm in range(self.bandit.nbr_arms)}
        self.last_best_mean = {arm: 0 for arm in range(self.bandit.nbr_arms)}
        self.best_arm = 0
        self.first_time = True
        self.rng = np.random.default_rng()

    def specific_update(self, arm, r):
        self.samples[arm].append(r)

    def compute_indices(self):
        max_mean = np.max(self.means)
        if self.all_selected:
            if self.first_time:
                self.best_arm = np.argmax(self.means)
                self.first_time = False
                for arm in range(self.bandit.nbr_arms):
                    m = self.means[arm]
                    n = self.nbr_pulls[arm]
                    samples = self.samples[arm]
                    res = kinf(samples, max_mean, self.upper_bound)
                    self.kinf[arm] = res
                    self.last_best_mean[arm] = max_mean
                    if np.isclose(m, max_mean):
                        self.indices[arm] = 0.
                    else:
                        self.indices[arm] = n * res.fun
            else:
                if np.argmax(self.means) != self.best_arm:
                    self.best_arm = np.argmax(self.means)
                    for arm in range(self.bandit.nbr_arms):
                        m = self.means[arm]
                        n = self.nbr_pulls[arm]
                        samples = self.samples[arm]
                        res = kinf(samples, max_mean, self.upper_bound)
                        self.kinf[arm] = res
                        self.last_best_mean[arm] = max_mean
                        if np.isclose(m, max_mean):
                            self.indices[arm] = 0.
                        else:
                            self.indices[arm] = n * res.fun
                else:
                    for arm in range(self.bandit.nbr_arms):
                        m = self.means[arm]
                        n = self.nbr_pulls[arm]
                        if arm == self.last_played_arm:
                            if np.isclose(m, max_mean):
                                self.indices[arm] = 0.
                            else:
                                samples = self.samples[arm]
                                res = kinf(samples, max_mean, self.upper_bound)
                                self.kinf[arm] = res
                                self.last_best_mean[arm] = max_mean
                                self.indices[arm] = n * res.fun
                        else:
                            if np.isclose(m, max_mean):
                                self.indices[arm] = 0.
                            else:
                                res = self.kinf[arm]
                                approx_kinf = - res.fun + (max_mean - self.last_best_mean[arm]) * res.x
                                self.indices[arm] = - n * approx_kinf
        else:
            self.indices = self.nbr_pulls.astype(float)
            self.best_arm = np.argmax(self.means)

    def choose_an_arm(self):
        if self.all_selected:
            g = self.rng.gumbel(0, 1, self.bandit.nbr_arms)
            arm = randamax(g + self.indices)
        else:
            arm = randamin(self.indices)
        return arm


########################################
#                OMED                 #
########################################
class OMED(OnlineLearningAlgorithm):
    def __init__(self, bandit, name="OMED", eta=lambda n: np.sqrt(np.log(2) / (4 * n)), gamma=0.001, upper_bound=1.):
        OnlineLearningAlgorithm.__init__(self, bandit, name)
        self.name = name
        self.eta = eta
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.ub = upper_bound + gamma
        self.kl_ub = lambda x: np.log(1. / (self.ub - x))

        self.z = np.zeros(self.bandit.nbr_arms, dtype=int)
        self.nbr_pulls_lc = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float)
        self.lbd = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms)) + 0.5
        self.kl = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms))
        self.queue = [arm for arm in range(self.bandit.nbr_arms)]
        self.indices = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float) + self.initial_index
        self.leader = None

        self.specific_reset()

    def specific_reset(self):
        self.z = np.zeros(self.bandit.nbr_arms, dtype=int)
        self.nbr_pulls_lc = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float)
        self.lbd = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms)) + 0.5
        self.kl = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms))
        self.queue = [arm for arm in range(self.bandit.nbr_arms)]
        self.indices = np.zeros((self.bandit.nbr_arms, self.bandit.nbr_arms), dtype=float) + self.initial_index
        self.leader = None

    def compute_indices(self):
        if self.leader is None:
            self.leader = randamax(self.nbr_pulls, self.means)
        if self.z[self.last_played_arm] == 1:
            self.z[self.last_played_arm] = 0
            self.nbr_pulls_lc[self.leader, self.last_played_arm] += 1

            leader = self.leader

            leader_mean = self.means[leader]
            arm = self.last_played_arm
            r = self.last_reward

            lbd = self.lbd[leader, arm]
            n = self.nbr_pulls_lc[leader, arm]

            eta = self.eta(n) / (1 + self.eta(n))
            eta_plus_un = self.eta(n + 1) / (1 + self.eta(n + 1))
            frac = (self.ub - r) / (self.ub - leader_mean)
            lbd = lbd * (1 - eta + eta * frac / (1 - lbd + lbd * frac))
            lbd = lbd * (eta_plus_un / eta) + (1 - (eta_plus_un / eta)) * 0.5
            self.lbd[leader, arm] = lbd

            if self.means[arm] >= leader_mean:
                self.kl[leader, arm] = 0.
            else:
                self.kl[leader, arm] += np.log(1 - lbd + lbd * frac)
                self.kl[leader, arm] = np.minimum(self.kl[leader, arm], n * self.kl_ub(leader_mean))
                self.kl[leader, arm] = np.maximum(0., self.kl[leader, arm])
            self.indices[leader, arm] = self.kl[leader, arm]

    def duel(self):
        leader = randamax(self.nbr_pulls, self.means)
        self.leader = leader
        leader_mean = self.means[leader]
        n = self.nbr_pulls[leader]
        pull_threshold = 10 * np.sqrt(n)

        for challenger in range(self.bandit.nbr_arms):
            if challenger != leader:
                if (self.nbr_pulls_lc[leader, challenger] >= pull_threshold) or (leader_mean >= self.upper_bound) or (
                        self.means[challenger] >= self.upper_bound):
                    if self.means[challenger] >= leader_mean:
                        self.queue.append(challenger)
                else:
                    idx = self.indices[leader, challenger]
                    if idx >= 30:
                        p = np.exp(-30)
                    elif idx <= 1e-9:
                        p = 1
                    else:
                        p = np.exp(-idx)

                    if np.random.binomial(1, p) == 1:
                        self.queue.append(challenger)
                        self.z[challenger] = 1

        if len(self.queue) == 0:
            self.queue.append(leader)

    def choose_an_arm(self):
        if len(self.queue) == 0:
            self.duel()
            arm = self.queue.pop()
        else:
            arm = self.queue.pop()
        return arm
