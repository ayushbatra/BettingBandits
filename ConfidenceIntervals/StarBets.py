from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
import numpy as np
from array import array
from ConfidenceIntervals.vendor.starBets.STaRbets.star import star

class StarCI(ConfidenceInterval):
    def __init__(self):
        super().__init__()
        self.num_rounds = 0
        self.observations = array('d')

    def observe(self, observation: float) -> None:
        assert 0 <= observation <= 1, "observation must be in [0, 1]"
        self.num_rounds += 1
        self.observations.append(observation)

    def upper_bound(self, alpha: float) -> float:
        assert 0 < alpha < 1, "alpha must be between 0 and 1"
        if self.num_rounds == 0:
            return 1.0

        x_array = np.frombuffer(self.observations, dtype=np.float64)
        try:
            upper = 1.0 - star(1.0 - x_array, alpha=alpha)
            return float(min(1.0, upper))
        finally:
            del x_array

