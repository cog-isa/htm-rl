from typing import Optional

from htm_rl.common.utils import lin_sum


class MovingAverageTracker:
    short_ma: Optional[float]
    short_ma_learning_rate: float

    long_ma: Optional[float]
    long_ma_learning_rate: float

    def __init__(self, short_ma_learning_rate: float, long_ma_learning_rate: float):
        self.short_ma = None
        self.short_ma_learning_rate = short_ma_learning_rate
        self.long_ma = None
        self.long_ma_learning_rate = long_ma_learning_rate

    def update(self, value: float):
        if self.short_ma is None:
            self.short_ma = self.long_ma = value
        else:
            self.short_ma = lin_sum(self.short_ma, self.short_ma_learning_rate, value)
            self.long_ma = lin_sum(self.long_ma, self.long_ma_learning_rate, value)
