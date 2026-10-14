# common/distributions.py
from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    def __init__(self, seed=None, divisions=None):
        assert divisions is None or (isinstance(divisions, int) and divisions > 0), "divisions must be a positive integer"
        self.rng = np.random.default_rng(seed)
        self.divisions = divisions

    @abstractmethod
    def _sample(self) -> float:
        """Draw one observation."""
        pass

    def next_observation(self) -> float:
        x = float(self._sample())

        assert 0 <= x <= 1, "Observation must be in [0, 1]"

        if self.divisions is not None:
            x = round(x * self.divisions) / self.divisions

        return x

    @abstractmethod
    def mean(self) -> float:
        """Expected value returned by next_observation()."""
        pass

