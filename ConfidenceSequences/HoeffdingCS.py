from ConfidenceSequences.confidence_sequences import ConfidenceSequence
import numpy as np

class HoeffdingCS(ConfidenceSequence):
    def __init__(self, divisions: int | None = None):
        super().__init__(divisions=divisions)
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
        log_term = np.log(n) + np.log(n + 1) - np.log(alpha)
        radius = np.sqrt(log_term / (2 * n))

        return float(np.minimum(1.0, mean + radius))














