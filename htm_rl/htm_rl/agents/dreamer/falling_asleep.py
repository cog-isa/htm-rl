from htm_rl.agents.q.balancing_param import BalancingParam
from htm_rl.common.utils import DecayingValue


class TdErrorBasedFallingAsleep:
    boost_prob_alpha: DecayingValue
    prob_threshold: float

    def __init__(
            self, boost_prob_alpha: DecayingValue, prob_threshold: float
    ):
        self.boost_prob_alpha = boost_prob_alpha
        self.prob_threshold = prob_threshold


class AnomalyBasedFallingAsleep:
    anomaly_threshold: BalancingParam
    probability: BalancingParam
    breaking_point: float
    power: float

    def __init__(
            self, anomaly_threshold: list[float], probability: list[float],
            breaking_point: float, power: float,
    ):
        self.anomaly_threshold = BalancingParam(*anomaly_threshold)
        self.probability = BalancingParam(*probability)
        # from anomaly- to recall-based
        self.breaking_point = 1 - breaking_point
        self.power = power
