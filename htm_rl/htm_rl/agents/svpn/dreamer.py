from typing import Tuple, Optional

import numpy as np

from htm_rl.agents.svpn.sparse_value_network import SparseValueNetwork
from htm_rl.agents.ucb.sparse_value_network import modify_factor_tuple
from htm_rl.common.sdr import SparseSdr


class Dreamer:
    svn: SparseValueNetwork

    enabled: bool
    dreaming_prob_alpha: Tuple[float, float]
    dreaming_prob_threshold: float
    rnd_move_prob: float
    learning_rate: Tuple[float, float]
    learning_rate_factor: float
    td_lambda: bool
    trace_decay: float
    nest_traces: bool
    cell_eligibility_trace: Optional[np.ndarray]
    starting_sa_sdr: Optional[SparseSdr]
    TD_error: Optional[float]

    def __init__(
            self, svn: SparseValueNetwork,
            dreaming_prob_alpha: Tuple[float, float], dreaming_prob_threshold: float,
            rnd_move_prob: float,
            learning_rate_factor: float, trace_decay: Optional[float],
            enabled: bool = True, nest_traces: bool = True
    ):
        if trace_decay is None:
            trace_decay = svn.trace_decay

        self.enabled = enabled
        self.dreaming_prob_alpha = dreaming_prob_alpha
        self.dreaming_prob_threshold = dreaming_prob_threshold
        self.rnd_move_prob = rnd_move_prob
        self.learning_rate_factor = learning_rate_factor
        self.learning_rate = svn.learning_rate
        self.td_lambda = trace_decay > .0
        self.trace_decay = trace_decay
        self.nest_traces = nest_traces

        self.svn = svn
        self.cell_eligibility_trace = None
        self.starting_sa_sdr = None
        self.TD_error = None

    def put_into_dream(self, starting_sa_sdr):
        wake_svn = self.svn
        self.learning_rate = wake_svn.learning_rate
        wake_svn.learning_rate = modify_factor_tuple(wake_svn.learning_rate, self.learning_rate_factor)
        wake_svn.trace_decay, self.trace_decay = self.trace_decay, wake_svn.trace_decay

        self.cell_eligibility_trace = wake_svn.cell_eligibility_trace.copy()
        if not self.nest_traces:
            wake_svn.cell_eligibility_trace.fill(.0)

        self.starting_sa_sdr = starting_sa_sdr.copy()
        self.TD_error = wake_svn.TD_error

    def reset_dreaming(self, i_rollout=None):
        dreaming_svn = self.svn
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()
        if not self.nest_traces:
            dreaming_svn.cell_eligibility_trace.fill(.0)
        if i_rollout is not None:
            dreaming_svn.learning_rate = modify_factor_tuple(
                dreaming_svn.learning_rate,
                1.0/(i_rollout + 1.)**.5
            )
        return self.starting_sa_sdr.copy()

    def wake(self):
        dreaming_svn = self.svn
        dreaming_svn.learning_rate = self.learning_rate
        dreaming_svn.trace_decay, self.trace_decay = self.trace_decay, dreaming_svn.trace_decay
        dreaming_svn.TD_error = self.TD_error
        dreaming_svn.cell_eligibility_trace = self.cell_eligibility_trace.copy()