class BalancingParam:
    value: float
    min_value: float
    max_value: float

    def __init__(self, initial_value: float, min_value: float, max_value: float):
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value

    def add(self, delta):
        self.update(self.value + delta)

    def scale(self, factor):
        self.update(self.value * factor)

    def update(self, new_value):
        if new_value > self.max_value:
            new_value = self.max_value
        elif new_value < self.min_value:
            new_value = self.min_value

        self.value = new_value
