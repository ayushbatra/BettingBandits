
import numpy as np
from Algorithms.bandit_algo import BanditAlgorithm
from ConfidenceIntervals.confidence_intervals import ConfidenceInterval
from collections.abc import Callable


class UCB1(BanditAlgorithm):
    def __init__(self, n_arms: int, ci_class: type[ConfidenceInterval],horizon: int | None = None, alpha: Callable[[int], float] | None = None,):
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.intervals = [ci_class() for _ in range(n_arms)]
        self.counts = np.zeros(n_arms, dtype=int)
        self.rounds_completed = 0
        self.ci_class = ci_class
        self.alpha_schedule = (
            alpha if alpha is not None else lambda t: 1.0 / (n_arms * t**4)
        )

    def alpha(self) -> float:
        t = self.rounds_completed + 1
        value = float(self.alpha_schedule(t))
        if not 0.0 < value < 1.0:
            raise ValueError(f"alpha({t}) must be in (0, 1), got {value}")
        return value

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

