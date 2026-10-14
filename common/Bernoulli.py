import numpy as np
from common.distributions import Distribution

class Bernoulli(Distribution):
    def __init__(self, p, seed=None, divisions=None):
        super().__init__(seed=seed, divisions=divisions)
        assert 0 <= p <= 1, "p must be between 0 and 1"
        self.p = p

    def _sample(self) -> float:
        return float(self.rng.binomial(1, self.p))

    def mean(self) -> float:
        return self.p