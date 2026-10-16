from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
import numpy as np
from array import array

from ConfidenceIntervals.vendor.confseq.src.confseq.betting import betting_cs
from ConfidenceIntervals.vendor.confseq.src.confseq.betting_strategies import lambda_aKelly


class Grapa(ConfidenceInterval):
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

        x_array = np.frombuffer(self.observations, dtype=np.float64).copy()
        try:
            _, upper = betting_cs(
                x_array,
                alpha=alpha,
                lambdas_fns_positive=[
                    lambda x, m: lambda_aKelly(x, np.clip(m, 1e-10, 1 - 1e-10))
                ],
                theta=1,
                trunc_scale=0.5,
            )
            return float(upper[-1])
        finally:
            del x_array