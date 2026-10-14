from arms import *
from utils import kinf
import pickle


########################################
#            Bandit Class              #
########################################
class Bandit:
    def __init__(self, arms, name="bandit", upper_bound=1.):
        self.arms = arms
        self.nbr_arms = len(arms)
        self.name = name
        self.ub = upper_bound

        # Preprocessing of useful statistics

        # Expected value of arms
        # Bandits community vocabulary
        self.rewards = np.array([arms[i].mean for i in range(self.nbr_arms)])
        # Probability community vocabulary
        self.means = self.rewards

        # Best arm index (one of) and expected value (unique)
        self.best_arm = np.argmax(self.rewards)
        self.best_reward = np.max(self.rewards)
        self.best_mean = self.best_reward

        # Regret/suboptimality gap of arms
        self.regrets = self.best_reward - self.rewards

    def __str__(self):
        return f"Bandit({self.arms})"

    def __repr__(self):
        return f"Bandit({self.arms})"

    # Bandits community vocabulary
    def pull(self, arm):
        return self.arms[arm].sample()

    # Probability community vocabulary
    def sample(self, idx):
        return self.pull(idx)

    def complexity(self):
        suboptimal_arms = np.where(self.regrets != 0.)[0]
        term_1 = self.regrets[suboptimal_arms]
        term_2 = []
        for i in range(self.nbr_arms):
            samples = [self.arms[i].sample() for _ in range(200)]
            term_2.append(-kinf(samples, self.best_reward).fun)
        c = sum(term_1 / term_2)
        return c


########################################
#          Bandit Instances            #
########################################

class BernoulliBandit(Bandit):
    def __init__(self, means):
        assert len(means) > 0, "means should not be empty"
        assert np.all(means >= 0) and np.all(means <= 1), \
            "Bernoulli mean should be between 0 and 1:\n(means={means})"
        arms = [Bernoulli(m) for m in means]
        Bandit.__init__(self, arms, name="Bernoulli")


class TruncatedExpBandit(Bandit):
    def __init__(self, scales):
        assert len(scales) > 0, "scales should not be empty"
        arms = [TruncExp(m) for m in scales]
        Bandit.__init__(self, arms, name="TruncatedExp")


class BetaBandit(Bandit):
    def __init__(self, means, size=5):
        assert len(means) > 0, "means should not be empty"
        assert np.all(means >= 0) and np.all(means <= 1), \
            "Bernoulli mean should be between 0 and 1:\n(means={means})"
        arms = [Beta(m, size) for m in means]
        Bandit.__init__(self, arms, name="Beta")


class DssatBandit(Bandit):
    def __init__(self, file="dssat.pkl", rng=np.random.default_rng(), normalize=True):
        f = open(file, "rb")
        # write the python object (dict) to pickle file
        distributions = pickle.load(f)
        # close file
        f.close()

        samples = np.array(distributions['samples'])
        if normalize:
            lower_bound = np.min(samples, axis=1, keepdims=True)
            upper_bound = np.max(samples)
            samples = (samples - lower_bound) / (upper_bound - lower_bound)
            upper_bound = 1.
        else:
            upper_bound = np.max(samples)
        self.samples = samples
        arms = [Empirical(sample_array, rng=rng) for sample_array in samples]
        Bandit.__init__(self, arms, name="DSSAT", upper_bound=upper_bound)

    def reseed(self):
        for arm in self.arms:
            arm.rng = np.random.default_rng(seed=np.random.randint(0, 10000))
