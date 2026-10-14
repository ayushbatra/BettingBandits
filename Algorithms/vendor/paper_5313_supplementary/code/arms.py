import numpy as np
from scipy.stats import truncexpon, truncnorm, beta

########################################
#            Bounded Arms              #
########################################


class Bernoulli:
    def __init__(self, mean):
        self.name = 'Bernoulli'
        # assert 0 <= mean <= 1, f"The mean of a Bernoulli should between 0 and 1: mean={mean}"
        self.mean = mean
        self.std = mean * (1 - mean)

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return np.random.binomial(1, self.mean)


class TruncExp:
    def __init__(self, scale):
        self.mean = truncexpon.stats(scale, scale=1/scale, moments='m')
        self.std = truncexpon.stats(scale, scale=1/scale, moments='s')
        self.name = "Truncated Exponential"
        self.scale = scale

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return truncexpon.rvs(self.scale, scale=1/self.scale)


class Beta:
    def __init__(self, mean, size=5):
        self.mean = mean
        self.a = self.mean * size
        self.b = (1 - self.mean) * size
        self.std = beta.stats(self.a, self.b, moments='s')
        self.name = "Beta"

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return beta.rvs(self.a, self.b)


class TruncNorm:
    def __init__(self, mean, std=1.):
        self.mean = mean
        self.std = std
        self.name = "Normal (truncated)"

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return truncnorm.rvs(loc=self.mean, a=-20, b=1-self.mean-1e-9)


class Empirical:
    def __init__(self, sample_array, rng, name="Empirical"):
        self.sample_array = sample_array
        self.mean = np.mean(sample_array)
        self.std = np.std(sample_array)
        self.rng = rng
        self.name = name

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        # return self.rng.choice(self.sample_array)
        return np.random.choice(self.sample_array)


########################################
#           Unbounded Arms             #
########################################


class Normal:
    def __init__(self, mean, std=1.):
        self.mean = mean
        self.std = std
        self.name = "Normal"

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f}, std={self.std:.3f})"

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)


class Poisson:
    def __init__(self, mean, std=1.):
        self.mean = mean
        self.std = std
        self.name = "Normal"

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f}, std={self.std:.3f})"

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)
