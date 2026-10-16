from Algorithms import UCB1, KLinfUCB, IMED, ThompsonSampling

from ConfidenceIntervals import HoeffdingCI, PredictablePluginEB, PreciseCI, HedgedCI, StarCI, UniversalPortfolioCI, Grapa

from common import Bernoulli, Beta, TwoPoint

from Environment import Environment

if __name__ == "__main__":
    environment = Environment(
        horizon=10**2,
        algorithm=UCB1,
        algorithm_params={"ci_class": Grapa},
        arm_classes=[Bernoulli, Beta],
        arm_params=[
            {"p": 0.6, "seed": 1},
            {"alpha": 2, "beta": 3, "seed": 2},
        ],
    )

    result = environment.run()

    print("Regret:", result["regret"])
    print("Pull counts:", result["pulls"])
    print("Pull frequencies:", result["frequencies"])