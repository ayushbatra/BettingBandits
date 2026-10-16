from Algorithms.UCB1 import UCB1
from Algorithms.KLinfUCB import KLinfUCB
from Algorithms.Imed import IMED
from Algorithms.ThompsonSampling import ThompsonSampling
from ConfidenceIntervals.HoeffdingCI import HoeffdingCI
from ConfidenceIntervals.Precise import PreciseCI
from ConfidenceIntervals.PredictablePluginEB import PredictablePluginEB
from ConfidenceIntervals.HedgedCI import HedgedCI
from ConfidenceIntervals.StarBets import StarCI
from common.Bernoulli import Bernoulli
from common.Beta import Beta
from Environment.Environment import Environment

if __name__ == "__main__":
    environment = Environment(
        horizon=10**3,
        algorithm=UCB1,
        algorithm_params={"ci_class": PreciseCI},
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