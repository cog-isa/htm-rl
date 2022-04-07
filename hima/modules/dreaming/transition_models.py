from hima.modules.dreaming.sa_encoder import SaEncoder
from hima.modules.dreaming.transition_model import TransitionModel
from hima.common.sdr import SparseSdr
from hima.modules.htm.temporal_memory import ClassicTemporalMemory as TemporalMemory


class SaTransitionModel(TransitionModel):
    def __init__(self, sa_encoder: SaEncoder, tm: dict):
        tm = self.make_tm(sa_encoder, tm)
        super(SaTransitionModel, self).__init__(tm)

    def process(self, s_a: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        return super(SaTransitionModel, self).process(s_a, learn)

    @staticmethod
    def make_tm(sa_encoder, tm: dict) -> TemporalMemory:
        from hima.modules.dreaming.sa_encoders import SpSaEncoder

        if isinstance(sa_encoder, SpSaEncoder):
            s_cols = sa_encoder.state_sp.output_sdr_size
            a_cols = sa_encoder.action_encoder.output_sdr_size
            s_active_bits = sa_encoder.state_sp.n_active_bits
            a_active_bits = sa_encoder.action_encoder.n_active_bits
            n_columns = s_cols + a_cols
            n_active_bits = s_active_bits + a_active_bits
        else:
            raise ValueError()

        return TemporalMemory(
            n_columns=n_columns, n_active_bits=n_active_bits,
            **tm
        )


def make_transition_model(sa_encoder: SaEncoder, transition_model_config):
    return SaTransitionModel(sa_encoder=sa_encoder, **transition_model_config)
