import numpy as np
from common.distributions import Distribution


class Beta(Distribution):
    def __init__(self, alpha, beta, seed=None):
        super().__init__(seed=seed)
        assert np.isfinite(alpha) and alpha > 0 and np.isfinite(beta) and beta > 0, "alpha and beta must be positive"
        self.alpha = alpha
        self.beta = beta

    def _sample(self) -> float:
        return float(self.rng.beta(self.alpha, self.beta))

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
