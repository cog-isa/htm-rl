from htm_rl.agents.q.qvn import QValueNetwork
from htm_rl.common.utils import modify_factor_tuple


class QValueNetworkDouble(QValueNetwork):
    origin_learning_rate: tuple[float, float]
    learning_rate_factor: float

    # noinspection PyMissingConstructor
    def __init__(self, origin: QValueNetwork, learning_rate_factor: float):
        self.origin_learning_rate = origin.learning_rate
        self.learning_rate_factor = learning_rate_factor
        self.learning_rate = modify_factor_tuple(origin.learning_rate, learning_rate_factor)
        self.discount_factor = origin.discount_factor
        self.cell_value = origin.cell_value
        self.last_td_error = 0.
