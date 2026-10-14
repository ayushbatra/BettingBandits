import numpy as np
from common.distributions import Distribution

class TwoPoint(Distribution):
    def __init__(self, a, b, weight_a, weight_b, seed=None):
        super().__init__(seed=seed)
        total = weight_a + weight_b
        assert 0 <= a <= 1 and 0 <= b <= 1, "a and b must be in [0, 1]"
        assert np.isfinite(weight_a) and weight_a >= 0 and np.isfinite(weight_b) and weight_b >= 0 and np.isfinite(total) and total > 0, "Weights must be finite, nonnegative, with positive finite sum"
        total = weight_a + weight_b
        self.a = a
        self.b = b
        self.prob_a = weight_a / total

    def _sample(self) -> float:
        return self.a if self.rng.random() < self.prob_a else self.b

    def mean(self) -> float:
        return self.prob_a * self.a + (1 - self.prob_a) * self.b
