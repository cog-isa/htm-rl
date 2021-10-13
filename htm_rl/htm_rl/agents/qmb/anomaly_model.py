import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_slice_lin_sum, exp_decay


class AnomalyModel:
    """
    Represents learned model of the anomaly for the transition
    model on columns-level.
    An anomaly of the transition (s, a) --> s' is stored in the corresponding
    ndarray of shape (state_space, action_space) by using (s', a) as the key.
    I.e. `anomaly[s][a]` corresponds to an anomaly of getting to the state s
    by doing action a from the previous state. Such form is equivalent to
    using (s, a) but is more convenient.
    """
    anomaly: np.ndarray
    learning_rate: tuple[float, float]
    last_error: float

    def __init__(self, cells_sdr_size, n_actions: int, learning_rate: tuple[float, float]):
        self.learning_rate = learning_rate
        self.anomaly = np.ones((cells_sdr_size, n_actions), dtype=np.float)
        self.last_error = 0.

    def update(self, prev_action: int, s: SparseSdr, anomaly: float):
        lr = self.learning_rate[0]
        # anomaly_estimate = self.state_anomaly(s, prev_action)
        anomaly_estimate = self.anomaly[s, prev_action]
        error = anomaly - anomaly_estimate

        self.anomaly[s, prev_action] += lr * error
        self.last_error = error.mean()

    def state_anomaly(self, s: SparseSdr, prev_action: int = None):
        if len(s) == 0:
            return 1.

        # reduce by state encoding
        action_anomaly = np.median(self.anomaly[s], axis=0)
        if prev_action is not None:
            return action_anomaly[prev_action]

        # reduce by actions
        anomaly = np.median(action_anomaly)
        return anomaly

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)
