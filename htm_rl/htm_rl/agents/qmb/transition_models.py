from htm.bindings.sdr import SDR

from htm_rl.agents.q.sa_encoders import SpSaEncoder, CrossSaEncoder
from htm_rl.common.sdr import SparseSdr
from htm_rl.common.sdr_encoders import SdrConcatenator
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


class TransitionModel:
    tm: TemporalMemory

    anomaly: float
    precision: float
    recall: float

    _proximal_input_sdr: SDR    # cached SDR
    _predicted_columns_sdr: SDR   # cached SDR
    _tm_dump: bytes

    def __init__(self, tm: TemporalMemory):
        self.tm = tm
        self.precision = 0.
        self.recall = 0.
        self.anomaly = 1.

        self._proximal_input_sdr = SDR(self.tm.n_columns)
        self._predicted_columns_sdr = SDR(self.tm.n_columns)

    def reset(self):
        self.tm.reset()
        self._predicted_columns_sdr.sparse = []

    def preprocess(self, *data) -> SparseSdr:
        if len(data) == 1 and isinstance(data[0], SparseSdr):
            return data[0]
        raise NotImplementedError()

    def process(self, proximal_input: SparseSdr, learn: bool) -> tuple[SparseSdr, SparseSdr]:
        """
        Given new piece of proximal input data, processes it by sequentially activating cells
        and then depolarizes them, making prediction about next proximal input.

        :param proximal_input: sparse SDR of column activations.
        :param learn: whether or not to force learning.
        :return: tuple (active cells, depolarized cells) of sparse SDRs.
        """
        active_cells = self._activate_cells(proximal_input, learn)
        depolarized_cells = self._depolarize_cells(learn)
        return active_cells, depolarized_cells

    def _activate_cells(self, proximal_input: SparseSdr, learn: bool) -> SparseSdr:
        """
        Given proximal input SDR, activates TM cells and [optionally] applies learning step.

        Learning is applied to already made predictions - to active segments of depolarized cells.

        :param proximal_input: sparse SDR of active columns.
        :param learn: whether or not to force learning.
        :return: sparse SDR of active cells.
        """
        self._proximal_input_sdr.sparse = proximal_input
        self._compute_prediction_quality()
        self.tm.activateCells(self._proximal_input_sdr, learn=learn)
        return self.tm.getActiveCells().sparse

    def _depolarize_cells(self, learn: bool) -> SparseSdr:
        """
        Given the current state of active cells, activates cells' segments
        leading to cells depolarization.

        :param learn: whether or not TM should make a learning step too.
        :return: sparse SDR of depolarized cells
        """
        self.tm.activateDendrites(learn=learn)
        predicted_cells = self.tm.getPredictiveCells().sparse
        self._predicted_columns_sdr.sparse = list(self.columns_from_cells(predicted_cells))
        return predicted_cells

    @property
    def predicted_cols(self) -> SparseSdr:
        return self._predicted_columns_sdr.sparse

    def columns_from_cells(self, cells_sparse_sdr: SparseSdr) -> SparseSdr:
        """
        Converts cells sparse SDR to columns sparse SDR.

        :param cells_sparse_sdr: cells sparse SDR
        :return: columns sparse SDR
        """
        cpc = self.tm.cells_per_column
        return set(cell_ind // cpc for cell_ind in cells_sparse_sdr)

    def _compute_prediction_quality(self):
        n_predicted = self._predicted_columns_sdr.getSum()
        n_active = self._proximal_input_sdr.getSum()
        n_overlap = self._predicted_columns_sdr.getOverlap(self._proximal_input_sdr)

        precision = n_overlap / n_predicted if n_predicted > 0 else .0
        recall = n_overlap / n_active if n_active > 0 else .0
        self.precision, self.recall = precision, recall

        # recall importance factor (beta times)
        beta = 3
        if precision == .0 and recall == .0:
            f_beta_score = 0.
        else:
            f_beta_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        self.anomaly = 1. - f_beta_score

    def __getstate__(self):
        # used to pickle object
        data = (
            self.tm
        )
        return data

    def __setstate__(self, state):
        # used to unpickle
        self.tm = state


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
