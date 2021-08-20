from htm_rl.agents.q.sa_encoders import SpSaEncoder, CrossSaEncoder
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import SdrConcatenator
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class SsaTransitionModel(TransitionModel):
    s_sa_concatenator: SdrConcatenator

    def __init__(self, sa_encoder: SpSaEncoder, tm: dict):
        tm = self.make_tm(sa_encoder, tm)
        super(SsaTransitionModel, self).__init__(tm)

        self.s_sa_concatenator = SdrConcatenator([
            sa_encoder.state_sp,
            sa_encoder.sa_sp
        ])

    def preprocess(self, s: SparseSdr, sa: SparseSdr) -> SparseSdr:
        s_sa = self.s_sa_concatenator.concatenate(s, sa)
        return s_sa

    def process(self, s_sa: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        return super(SsaTransitionModel, self).process(s_sa, learn)

    @staticmethod
    def make_tm(sa_encoder, tm: dict) -> TemporalMemory:
        s_size = sa_encoder.state_sp.output_sdr_size
        sa_size = sa_encoder.sa_sp.output_sdr_size
        s_active_bits = sa_encoder.state_sp.n_active_bits
        sa_active_bits = sa_encoder.sa_sp.n_active_bits

        return TemporalMemory(
            n_columns=s_size + sa_size,
            n_active_bits=s_active_bits + sa_active_bits,
            **tm
        )


class SaTransitionModel(TransitionModel):
    def __init__(self, sa_encoder: CrossSaEncoder, tm: dict):
        tm = self.make_tm(sa_encoder, tm)
        super(SaTransitionModel, self).__init__(tm)

    def preprocess(self, s: SparseSdr, sa: SparseSdr) -> SparseSdr:
        return sa

    def process(self, sa: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        return super(SaTransitionModel, self).process(sa, learn)

    @staticmethod
    def make_tm(sa_encoder: CrossSaEncoder, tm: dict) -> TemporalMemory:
        return TemporalMemory(
            n_columns=sa_encoder.sa_encoder.output_sdr_size,
            n_active_bits=sa_encoder.sa_encoder.n_active_bits,
            **tm
        )
