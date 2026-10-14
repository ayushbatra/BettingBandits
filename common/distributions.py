# common/distributions.py
from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def _sample(self) -> float:
        """Draw one observation."""
        pass

    def next_observation(self) -> float:
        x = float(self._sample())
        assert 0 <= x <= 1, "Observation must be in [0, 1]"
        return x

    @abstractmethod
    def mean(self) -> float:
        """Expected value returned by next_observation()."""
        pass

