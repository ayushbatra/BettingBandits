from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
import numpy as np
from array import array

class PredictablePluginEB(ConfidenceInterval):
    def __init__(self):
        super().__init__()
        self.num_rounds = 0
        self.observations = array('d')

    @staticmethod
    def _Psi(lbd: float) -> float:
        return (-np.log1p(-lbd) - lbd) / 4

    def observe(self, observation: float) -> None:
        assert 0 <= observation <= 1, "observation must be in [0, 1]"
        self.observations.append(observation)
        self.num_rounds += 1

    def upper_bound(self, alpha: float) -> float:
        assert 0 < alpha < 1, "alpha must be between 0 and 1"

        n = self.num_rounds
        x_array = np.frombuffer(self.observations, dtype=np.float64)

        try:
            t_array = np.arange(1, n + 1, dtype=np.float64)
            log_term = np.log(2 / alpha)

            mu_array = (0.5 + np.cumsum(x_array)) / (t_array + 1)
            mu_prev_array = np.concatenate(([0.5], mu_array[:-1]))

            squared_errors = (x_array - mu_array) ** 2
            variance_prev = (0.25 + np.concatenate(([0.0], np.cumsum(squared_errors)[:-1]))) / t_array

            # Equation (15)
            lbd_array = np.minimum(
                np.sqrt( 2 * log_term / (variance_prev * t_array * np.log(1+t_array))),
                0.5
            )

            # Equation (14) -1
            v_array = 4 * (x_array - mu_prev_array) ** 2

            # Theorem 2: upper confidence sequence
            numerator = (
                np.sum(lbd_array * x_array)
                + np.sum(v_array * self._Psi(lbd_array))
                + log_term
            )
            denominator = np.sum(lbd_array)

            bound = float(min(1.0, numerator / denominator))

        finally:
            del x_array

        return bound


