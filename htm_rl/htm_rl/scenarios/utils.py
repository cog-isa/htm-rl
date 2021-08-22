class ProgressPoint:
    step: int
    episode: int

    def __init__(self):
        self.step = 0
        self.episode = 0

    @property
    def is_new_episode(self) -> bool:
        return self.step == 0

    def next_step(self):
        self.step += 1

    def end_episode(self):
        self.step = 0
        self.episode += 1


def filter_out_non_passable_items(config: dict, depth: int):
    """Recursively filters out non-passable args started with '.' and '_'."""
    if not isinstance(config, dict) or depth <= 0:
        return config

    return {
        k: filter_out_non_passable_items(v, depth - 1)
        for k, v in config.items()
        if not (k.startswith('.') or k.startswith('_'))
    }