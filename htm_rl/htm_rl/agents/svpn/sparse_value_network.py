from htm_rl.agents.ucb.sparse_value_network import SparseValueNetwork as SVN, exp_sum_update
from htm_rl.common.sdr import SparseSdr


class SparseValueNetwork(SVN):
    """SVN extended to support dreaming"""
    TD_error: float
    TD_error_decay: float

    def __init__(self, td_error_decay, **svn_args):
        super(SparseValueNetwork, self).__init__(**svn_args)
        self.TD_error = 0.
        self.TD_error_decay = td_error_decay

    # noinspection PyPep8Naming
    def _td_error(self, sa: SparseSdr, reward: float, sa_next: SparseSdr):
        TD_error = super(SparseValueNetwork, self)._td_error(sa, reward, sa_next)
        self.TD_error = exp_sum_update(self.TD_error, self.TD_error_decay, TD_error)
        return TD_error
