from array import array
import math
import numpy as np
from numba import njit
from ConfidenceIntervals.confidence_intervals import ConfidenceInterval


_GRID = np.linspace(0.0, 1.0, 10000)


@njit(cache=True)
def _upper_bound(xs, alpha, r, ms):
    n = len(xs)
    lga = np.log(1.0 / alpha)
    threshold = lga + np.log(r)

    # Scan in ascending order: only the first accepted point is needed.
    for m in ms:
        log_wealth = 0.0
        sg2 = 0.0

        for i in range(n):
            t = i + 0.0001
            x = 1.0 - xs[i]

            S = min(
                sg2 / t + (m + 0.0001) * n / t / t,
                0.0001 + m * (1.0 - m),
            )

            bet = math.sqrt(
                2.0 * max(lga - log_wealth, 0.0)
                / ((n - t) * S)
            )
            # The original lower clipping limit is negative;
            # bet is nonnegative, so only the upper limit matters.
            bet = min(bet, 1.0 / (m + 0.0001))

            diff = x - m
            log_wealth += math.log(1.0 + bet * diff)
            sg2 += diff * diff

        if log_wealth < threshold:
            lower = m - ms[1] + ms[0]
            return min(1.0, 1.0 - lower)

    return 0.0  # Original star() returns lower=1 if none accepted.


class OptimStarCI(ConfidenceInterval):
    def __init__(self):
        super().__init__()
        self.num_rounds = 0
        self.observations = array("d")

    def observe(self, observation: float) -> None:
        assert 0 <= observation <= 1
        self.observations.append(observation)
        self.num_rounds += 1

    def upper_bound(self, alpha: float) -> float:
        assert 0 < alpha < 1
        if self.num_rounds == 0:
            return 1.0

        xs = np.frombuffer(self.observations, dtype=np.float64)
        try:
            return float(_upper_bound(xs, alpha, np.random.rand(), _GRID))
        finally:
            del xs