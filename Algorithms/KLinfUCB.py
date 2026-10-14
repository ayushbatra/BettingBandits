"""

This uses implementation from

@article{baudry2023fast,
  title={Fast asymptotically optimal algorithms for non-parametric stochastic bandits},
  author={Baudry, Dorian and Pesquerel, Fabien and Degenne, R{\'e}my and Maillard, Odalric-Ambrym},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={11469--11514},
  year={2023}
}

"""


from Algorithms.bandit_algo import BanditAlgorithm
from Algorithms.vendor.paper_5313_supplementary.code.learning_algorithms import ( KLUCB as PaperKLinfUCB)


class KLinfUCB(BanditAlgorithm):
    def __init__(self, n_arms: int, horizon: int | None = None):
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.nbr_arms = n_arms
        self.policy = PaperKLinfUCB(self)

    def choose_next_arm(self) -> int:
        return int(self.policy.choose_an_arm())

    def observe_reward(self, arm: int, reward: float) -> None:
        assert 0 <= arm < self.n_arms, "Invalid arm index"
        assert 0 <= reward <= 1, "Reward must be in [0, 1]"
        self.policy.update_statistics(arm, float(reward))
