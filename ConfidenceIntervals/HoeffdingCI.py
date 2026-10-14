from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
import numpy as np

class HoeffdingCI(ConfidenceInterval):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.total = 0.0

    def observe(self, observation: float) -> None:
        assert 0 <= observation <= 1, "observation must be in [0, 1]"
        self.count += 1
        self.total += observation

    def upper_bound(self, alpha: float) -> float:
        assert 0 < alpha < 1, "alpha must be between 0 and 1"
        if self.count == 0:
            return 1.0
        n = self.count
        mean = self.total / n
        log_term =  - np.log(alpha)
        radius = np.sqrt(log_term / (2 * n))
        return float(np.minimum(1.0, mean + radius))

