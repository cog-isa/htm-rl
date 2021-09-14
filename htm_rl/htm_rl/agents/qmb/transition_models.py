from htm_rl.agents.q.sa_encoder import SaEncoder
from htm_rl.agents.q.sa_encoders import SpSaEncoder
from htm_rl.agents.qmb.transition_model import TransitionModel
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import SdrConcatenator
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class SaTransitionModel(TransitionModel):
    def __init__(self, sa_encoder: SaEncoder, tm: dict):
        tm = self.make_tm(sa_encoder, tm)
        super(SaTransitionModel, self).__init__(tm)

    def process(self, s_a: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        return super(SaTransitionModel, self).process(s_a, learn)

    @staticmethod
    def make_tm(sa_encoder, tm: dict) -> TemporalMemory:
        from htm_rl.agents.q.sa_encoders import SpSaEncoder, CrossSaEncoder

        if isinstance(sa_encoder, SpSaEncoder):
            s_cols = sa_encoder.state_sp.output_sdr_size
            a_cols = sa_encoder.action_encoder.output_sdr_size
            s_active_bits = sa_encoder.state_sp.n_active_bits
            a_active_bits = sa_encoder.action_encoder.n_active_bits
            n_columns = s_cols + a_cols
            n_active_bits = s_active_bits + a_active_bits
        elif isinstance(sa_encoder, CrossSaEncoder):
            n_columns = sa_encoder.sa_encoder.output_sdr_size
            n_active_bits = sa_encoder.sa_encoder.n_active_bits
        else:
            raise ValueError()

        return TemporalMemory(
            n_columns=n_columns, n_active_bits=n_active_bits,
            **tm
        )


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


def make_transition_model(sa_encoder: SaEncoder, transition_model_config):
    return SaTransitionModel(sa_encoder=sa_encoder, **transition_model_config)
