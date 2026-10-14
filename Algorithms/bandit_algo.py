from abc import ABC, abstractmethod

class BanditAlgorithm(ABC):
    def __init__(self, n_arms: int, horizon: int | None = None):
        assert isinstance(n_arms, int) and n_arms > 0, "n_arms must be a positive integer"
        assert horizon is None or (isinstance(horizon, int) and horizon > 0), "horizon must be a positive integer"

        self.n_arms = n_arms
        self.horizon = horizon

    @abstractmethod
    def choose_next_arm(self) -> int:
        """Return an arm index in range(self.n_arms)."""
        pass

    @abstractmethod
    def observe_reward(self, arm: int, reward: float) -> None:
        """Update the algorithm using the observed arm and reward."""
        pass









