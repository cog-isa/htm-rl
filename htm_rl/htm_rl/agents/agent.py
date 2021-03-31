from htm_rl.common.sdr import SparseSdr


class Agent:

    @property
    def name(self):
        raise NotImplementedError

    def act(self, reward: float, state: SparseSdr, first: bool):
        raise NotImplementedError
