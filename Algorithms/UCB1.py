
import numpy as np
from Algorithms.bandit_algo import BanditAlgorithm
from ConfidenceIntervals.confidence_intervals import ConfidenceInterval

class UCB1(BanditAlgorithm):
    def __init__(self, n_arms: int, ci_class: type[ConfidenceInterval],horizon: int | None = None):
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.intervals = [ci_class() for _ in range(n_arms)]
        self.counts = np.zeros(n_arms, dtype=int)
        self.rounds_completed = 0
        self.ci_class = ci_class

    def alpha(self) -> float:
        t = self.rounds_completed
        cs_alpha = getattr(self.ci_class, "alpha", None)
        if cs_alpha is not None:
            return cs_alpha(t)
        return 1.0 / (self.n_arms * t**4)

    def choose_next_arm(self,) -> int:
        if self.rounds_completed < self.n_arms:
            return self.rounds_completed        
        bounds = [cs.upper_bound(self.alpha()) for cs in self.intervals]
        return int(np.argmax(bounds))

    def observe_reward(self, arm: int, reward: float) -> None:
        assert 0 <= arm < self.n_arms, "Invalid arm index"
        self.intervals[arm].observe(reward)
        self.counts[arm] += 1
        self.rounds_completed += 1

