from Algorithms import UCB1, KLinfUCB, IMED, ThompsonSampling

from ConfidenceIntervals import HoeffdingCI, PredictablePluginEB, PreciseCI, HedgedCI, StarCI, UniversalPortfolioCI, Grapa, OptimStarCI

from common import Bernoulli, Beta, TwoPoint

from Environment import Environment

if __name__ == "__main__":
    environment = Environment(
        horizon=10**5,
        algorithm=ThompsonSampling,
        # algorithm_params={"ci_class": OptimStarCI, "alpha": lambda x: 1/x**1.1},
        arm_classes=[Bernoulli, Beta, TwoPoint],
        arm_params=[
            {"p": 0.4, "seed": 1},
            {"alpha": 2, "beta": 2, "seed": 2},
            {"a": 0.2, "b": 1, "weight_a": 1, "weight_b": 1, "seed": 2},
        ],
    )

    result = environment.run()

    print("Regret:", result["regret"])
    print("Pull counts:", result["pulls"])
    print("Pull frequencies:", result["frequencies"])

