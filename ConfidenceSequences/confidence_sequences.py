from abc import ABC, abstractmethod

class ConfidenceSequence(ABC):
    def __init__(self, divisions: int | None = None):
        assert divisions is None or (isinstance(divisions, int) and divisions > 0), "divisions must be a positive integer"
        self.divisions = divisions

    @abstractmethod
    def observe(self, observation: float) -> None:
        """Incorporate one observation from the declared support."""
        pass

    @abstractmethod
    def upper_bound(self, alpha: float) -> float:
        """
        Return the supremum of unrejected means in [0, 1].

        Require 0 < alpha < 1.
        Return 1.0 before any observations.

        For each fixed alpha, an implementation must guarantee:
            P(mean <= upper_bound(alpha) at every sample count)
            >= 1 - alpha.
        """
        pass






