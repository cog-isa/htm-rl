from htm_rl.agents.ucb.processing_unit import ProcessingUnit
from htm_rl.common.base_sa import SaRelatedComposition, Sa, SaSuperposition
from htm_rl.common.int_sdr_encoder import BitRange
from htm_rl.common.sdr import SparseSdr

SaShifts = SaRelatedComposition[int, int]
SaSplitSdr = SaRelatedComposition[SparseSdr, SparseSdr]


class UcbSaSdrEncoder(ProcessingUnit):
    """
    Composes 2 encoders for states and actions respectively.
    As a whole encodes/decodes sa values + adds some sa-related logic like:
        - get absolute indices range of action bits in SDR
    """
    value_bits: int
    total_bits: int
    activation_threshold: int

    _encoders: SaRelatedComposition
    _shifts: SaRelatedComposition

    def __init__(self, state_encoder, action_encoder):
        encoders = SaRelatedComposition(state_encoder, action_encoder)

        self.value_bits = sum(e.value_bits for e in encoders)
        self.total_bits = sum(e.total_bits for e in encoders)
        self.activation_threshold = sum(e.activation_threshold for e in encoders)

        self._shifts = self._get_shifts(encoders)
        self._encoders = encoders

    def encode(self, sa: Sa) -> SparseSdr:
        """
        Encodes SA to sparse SDR.
        """
        return [
            bit_index
            for x, encoder, shift in zip(sa, self._encoders, self._shifts)
            for bit_index in encoder.encode(x).shift(shift).unfold()
        ]

    def decode(self, sparse_sdr: SparseSdr) -> SaSuperposition:
        """
        Decodes sparse SDR into SA superposition, which is a SA pair of superpositions.

        Note that given SDR may contain multiple single values for each part of SA. That's why
        the result is not a single SA. It's also not a superposition of SAs, but a SA
        of superpositions. I.e. each part of SA is decoded separately as a superposition, which
        is a list of decoded [active] values.

        :param sparse_sdr: sparse SDR to decode
        :return: SA superposition. Each part of SA pair contains a list of decoded values.
        """

        split_sparse_sdrs = self.split_sa(sparse_sdr)
        state_superposition, action_superposition = (
            encoder.decode(sparse_sdr_)
            for encoder, sparse_sdr_ in zip(self._encoders, split_sparse_sdrs)
        )
        return SaRelatedComposition(state_superposition, action_superposition)

    @property
    def state_activation_threshold(self):
        return self._encoders.state.activation_threshold

    def states_indices_range(self) -> BitRange:
        """Gets sparse SDR indices range encoding states."""
        return BitRange(self._shifts.state, self._shifts.action)

    def actions_indices_range(self) -> BitRange:
        """Gets sparse SDR indices range encoding actions."""
        return BitRange(self._shifts.action, self.total_bits)

    def replace_action(self, sparse_sdr: SparseSdr, action: int) -> SparseSdr:
        """
        Gets sparse SDR which has action part replaced with specific action `action`.
        """
        action_only_sa = Sa(state=None, action=action)
        action_only_sdr = self.encode(action_only_sa)

        cell_range = self.actions_indices_range()
        no_action_sdr = [
            index for index in sparse_sdr if index not in cell_range
        ]

        replaced_action_sdr = no_action_sdr
        replaced_action_sdr.extend(action_only_sdr)
        return replaced_action_sdr

    def split_sa(self, sparse_sdr: SparseSdr) -> SaSplitSdr:
        """
        Splits given sparse SDR into pair of sparse SDRs, corresponding to each part of SA.

        I.e. sdr -> SA(state_sdr, action_sdr)
        :param sparse_sdr: sparse SDR encoding some SA
        :return: SA(state_sparse_sdr, action_sparse_sdr)
        """
        shifts = self._shifts
        split_indices = SaSplitSdr([], [])

        def put_for_corresponding_encoder(ind: int):
            if ind < shifts.action:
                split_indices.state.append(ind)
            else:
                split_indices.action.append(ind - shifts.action)

        for ind in sparse_sdr:
            put_for_corresponding_encoder(ind)
        return split_indices

    def format(self, sparse_sdr: SparseSdr, format_: str = None) -> str:
        """
        Formats sparse SDR to string with one of the supported formats.

        Supported formats are: 'full' and 'short'. If None then encoder's default is used.
        """

        split_sdrs = self.split_sa(sparse_sdr)
        return ' | '.join(
            encoder.format(sparse_sdr_, format_)
            for encoder, sparse_sdr_ in zip(self._encoders, split_sdrs)
        )

    @staticmethod
    def _get_shifts(encoders) -> SaRelatedComposition:
        actions_shift = encoders.state.total_bits
        return SaRelatedComposition(0, actions_shift)

    @property
    def input_shape(self):
        return (2, )

    @property
    def output_shape(self):
        return (self.total_bits, )

    def process(self, x):
        return self.encode(x)
