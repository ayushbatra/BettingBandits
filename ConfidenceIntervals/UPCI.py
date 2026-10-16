from array import array
import numpy as np

from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
from ConfidenceIntervals.vendor.CIGambling.CIG.methods.up import StockInvestmentCI


class UniversalPortfolioCI(ConfidenceInterval):
    def __init__(self):
        super().__init__()
        self.observations = array("d")
        self.model = StockInvestmentCI()

    def observe(self, observation: float) -> None:
        assert 0 <= observation <= 1, "observation must be in [0, 1]"
        self.observations.append(observation)

    def upper_bound(self, alpha: float) -> float:
        assert 0 < alpha < 1, "alpha must be between 0 and 1"
        if not self.observations:
            return 1.0

        x_array = np.frombuffer(self.observations, dtype=np.float64)
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                _, upper, *_ = self.model.construct(delta=alpha, xs=x_array, tqdm_=False)
            return float(upper[-1])
        finally:
            del x_array

##################################################################
## Caching Optimized implementation and choosing simpler design ##
##################################################################


# from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
# import numpy as np
# from scipy.special import betaln, logsumexp


# class UniversalPortfolioCI(ConfidenceInterval):
#     def __init__(self):
#         super().__init__()
#         self.num_rounds = 0
#         self.total = 0.0
#         self.log_coefficients = np.array([0.0])
#         self.log_weights = np.array([0.0])
#         self.indices = np.array([0.0])

#     def observe(self, observation: float) -> None:
#         assert 0 <= observation <= 1, "observation must be in [0, 1]"
#         x = float(observation)
#         old = self.log_coefficients

#         # Coefficients of product_i [(1 - x_i) + x_i * z].
#         if x == 0.0:
#             updated = np.concatenate((old, [-np.inf]))
#         elif x == 1.0:
#             updated = np.concatenate(([-np.inf], old))
#         else:
#             updated = np.logaddexp(
#                 np.concatenate((old + np.log1p(-x), [-np.inf])),
#                 np.concatenate(([-np.inf], old + np.log(x))),
#             )

#         self.num_rounds += 1
#         self.total += x
#         self.log_coefficients = updated

#         n = self.num_rounds
#         self.indices = np.arange(n + 1, dtype=float)

#         # Integrate against their Beta(1/2, 1/2) prior.
#         self.log_weights = (
#             updated
#             + betaln(self.indices + 0.5, n - self.indices + 0.5)
#             - betaln(0.5, 0.5)
#         )

#     def _log_wealth(self, mean: float) -> float:
#         n = self.num_rounds
#         k = self.indices
#         return float(logsumexp(
#             self.log_weights
#             - k * np.log(mean)
#             - (n - k) * np.log1p(-mean)
#         ))

#     def upper_bound(self, alpha: float) -> float:
#         assert 0 < alpha < 1, "alpha must be between 0 and 1"

#         if self.num_rounds == 0:
#             return 1.0

#         lower = self.total / self.num_rounds
#         upper = 1.0

#         if lower >= 1.0:
#             return 1.0

#         threshold = -np.log(alpha)

#         # At the empirical mean, UP wealth <= 1.
#         # Find the upper crossing of wealth = 1 / alpha.
#         while upper - lower > 1e-10:
#             midpoint = lower + (upper - lower) / 2

#             if self._log_wealth(midpoint) >= threshold:
#                 upper = midpoint
#             else:
#                 lower = midpoint

#         # Return the conservative side of the numerical bracket.
#         return float(upper)