import numpy as np
from tqdm.auto import tqdm
from Algorithms.bandit_algo import BanditAlgorithm
from common.distributions import Distribution


class Environment:
    def __init__(self, horizon: int, algorithm: type[BanditAlgorithm], arm_classes: list[type[Distribution]], arm_params: list[dict], algorithm_params: dict | None = None) -> None:
        assert horizon > 0, "horizon must be positive"
        assert arm_classes and len(arm_classes) == len(arm_params), "Provide one parameter dictionary per arm"

        self.horizon = horizon
        self.arms = [arm_class(**params) for arm_class, params in zip(arm_classes, arm_params)]
        self.algorithm = algorithm(n_arms=len(self.arms), horizon=horizon,**(algorithm_params or {}) )

    def run(self) -> dict:
        """Run once per Environment instance."""
        means = np.array([arm.mean() for arm in self.arms])
        chosen_arms = np.zeros(self.horizon, dtype=int)
        rewards = np.zeros(self.horizon)

        for t in tqdm(range(self.horizon), desc="Bandit simulation", unit="round"):
            arm = self.algorithm.choose_next_arm()
            reward = self.arms[arm].next_observation()
            self.algorithm.observe_reward(arm, reward)

            chosen_arms[t] = arm
            rewards[t] = reward

        pulls = np.bincount(chosen_arms, minlength=len(self.arms))
        regret_history = np.cumsum(means.max() - means[chosen_arms])

        return {
            "means": means,
            "chosen_arms": chosen_arms,
            "rewards": rewards,
            "pulls": pulls,
            "frequencies": pulls / self.horizon,
            "total_reward": float(rewards.sum()),
            "regret": float(regret_history[-1]),
            "regret_history": regret_history,
        }