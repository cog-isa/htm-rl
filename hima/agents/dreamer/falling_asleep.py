from hima.common.utils import DecayingValue


class TdErrorBasedFallingAsleep:
    boost_prob_alpha: DecayingValue
    prob_threshold: float

    def __init__(
            self, boost_prob_alpha: DecayingValue, prob_threshold: float
    ):
        self.boost_prob_alpha = boost_prob_alpha
        self.prob_threshold = prob_threshold


class AnomalyBasedFallingAsleep:
    anomaly_threshold: float
    alpha: float
    beta: float
    max_prob: float

    def __init__(
            self, anomaly_threshold: float, alpha: float,
            beta: float, max_prob: float,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.alpha = alpha
        self.beta = beta
        self.max_prob = max_prob
