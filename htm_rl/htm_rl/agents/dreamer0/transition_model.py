from htm_rl.common.sdr import SparseSdr
from htm_rl.htm_plugins.temporal_memory import TemporalMemory
from htm_rl.agents.qmb.transition_model import TransitionModel


class SaTransitionModel(TransitionModel):
    def __init__(self, sa_encoder, tm: dict):
        tm = TemporalMemory(
            n_columns=sa_encoder.output_sdr_size,
            n_active_bits=sa_encoder.n_active_bits,
            **tm
        )
        super(SaTransitionModel, self).__init__(tm)

    # noinspection PyMethodOverriding
    def process(self, s: SparseSdr, sa: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        return super(SaTransitionModel, self).process(sa, learn)
