from htm_rl.agents.q.cluster_memory_stats import ClusterMemoryStats
from htm_rl.common.utils import safe_divide


class DreamingStats:
    rollouts: int
    sum_depth: int

    wake_cluster_memory_stats: ClusterMemoryStats
    dreaming_cluster_memory_stats: ClusterMemoryStats

    def __init__(self, wake_cluster_memory_stats: ClusterMemoryStats = None):
        self.wake_cluster_memory_stats = wake_cluster_memory_stats
        if wake_cluster_memory_stats is not None:
            self.dreaming_cluster_memory_stats = ClusterMemoryStats()
        self.reset()

    def reset(self):
        self.rollouts = 0
        self.sum_depth = 0
        if self.wake_cluster_memory_stats is not None:
            self.wake_cluster_memory_stats.reset()
            self.dreaming_cluster_memory_stats.reset()

    def on_dreamed(self, rollouts: int, sum_depth: int):
        self.rollouts += rollouts
        self.sum_depth += sum_depth

    @property
    def avg_depth(self):
        return safe_divide(self.sum_depth, self.rollouts)
