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
