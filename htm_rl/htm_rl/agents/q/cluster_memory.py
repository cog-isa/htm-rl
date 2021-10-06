import numpy as np

from htm_rl.common.sdr import SparseSdr, DenseSdr
from htm_rl.common.utils import isnone


class ClusterMemory:
    trace_decay: float = .9

    sdr_size: int
    n_active_bits: int
    similarity_threshold: float
    similarity_threshold_delta: float
    similarity_threshold_limit: float

    n_clusters: int
    _clusters: np.ndarray
    _traces: np.ndarray
    _sizes: np.ndarray
    _clusters_trace: np.ndarray

    _cache_sdr: DenseSdr
    _cache_traces_sum: np.ndarray

    def __init__(
            self, sdr_size: int, n_active_bits: int,
            similarity_threshold: float, similarity_threshold_limit: float,
            max_n_clusters: int, max_tracked_bits_rate: float = 3.,
            similarity_threshold_delta: float = .0004
    ):
        self.sdr_size = sdr_size
        self.n_active_bits = n_active_bits
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_delta = similarity_threshold_delta
        self.similarity_threshold_limit = similarity_threshold_limit

        max_tracked_bits = int(n_active_bits * max_tracked_bits_rate)
        self.n_clusters = 0
        self._clusters = np.zeros((max_n_clusters, max_tracked_bits), dtype=np.int)
        self._traces = np.zeros_like(self._clusters, dtype=np.float)
        self._sizes = np.zeros(max_n_clusters, dtype=np.int)
        self._clusters_trace = np.zeros_like(self._sizes, dtype=np.float)

        self._cache_sdr = np.zeros(sdr_size, dtype=np.int)
        self._cache_traces_sum = np.zeros((max_n_clusters, 2), dtype=np.float)

    @property
    def empty(self):
        return self.n_clusters == 0

    @property
    def full(self):
        return self.n_clusters >= self._clusters.shape[0]

    @property
    def clusters(self):
        return self._clusters[:self.n_clusters]

    @property
    def traces(self):
        return self._traces[:self.n_clusters]

    @property
    def active_clusters(self):
        return self.clusters[:, -self.n_active_bits:]

    @property
    def active_clusters_trace(self):
        return self.traces[:, -self.n_active_bits:]

    def activate(
            self, sdr: SparseSdr, similarity: np.ndarray = None,
            matched_cluster: int = None
    ):
        if similarity is None:
            cluster, i_cluster, similarity = self.match(sdr)
            matched_cluster = i_cluster if cluster is not None else None

        if matched_cluster is None:
            matched_cluster = self.create(sdr)
        else:
            self.update(matched_cluster, sdr)

        return matched_cluster

    def create(self, sdr: SparseSdr):
        if self.full:
            self.clean_up()

        i_cluster = self.n_clusters
        n = len(sdr)

        self._clusters[i_cluster, -n:] = sdr
        self._traces[i_cluster, -n:] = 1.
        self._sizes[i_cluster] = n
        self.n_clusters += 1
        self._decay_cluster_traces(i_cluster)
        self._update_cluster_trace_sum_cache(i_cluster)
        return i_cluster

    def update(self, cluster: int, sdr: SparseSdr):
        size = self._sizes[cluster]
        # decay intra-cluster traces
        self._traces[cluster, -size:] *= self.trace_decay

        # noinspection PyTypeChecker
        sdr_set = set(sdr.tolist())
        # update traces for the intersection with sdr
        for k in range(1, size + 1):
            i = self._clusters[cluster, -k]
            if i in sdr_set:
                sdr_set.remove(i)
                self._traces[cluster, -k] += 1.

        to_add = list(sdr_set)
        max_cluster_size = self._clusters.shape[1]
        # append to cluster by increasing its size
        while to_add and size < max_cluster_size:
            size += 1
            self._clusters[cluster, -size] = to_add.pop()
            self._traces[cluster, -size] = 1.

        self._sizes[cluster] = size

        if to_add:
            # append to cluster by replacing elements with the lowest trace
            k = len(to_add)
            # the first k -- with the lowest trace
            partition = np.argpartition(self._traces[cluster], k, axis=-1)
            self._clusters[cluster, :] = self._clusters[cluster, partition]
            self._traces[cluster, :] = self._traces[cluster, partition]
            # replace with the new elements
            self._clusters[cluster, :k] = to_add
            self._traces[cluster, :k] = 1.

        valid_trace_part = self._traces[cluster, -size:]
        valid_cluster_part = self._clusters[cluster, -size:]

        # repartition to get active cluster on the right side
        partition = np.argpartition(valid_trace_part, -self.n_active_bits, axis=-1)
        self._traces[cluster, -size:] = valid_trace_part[partition]
        self._clusters[cluster, -size:] = valid_cluster_part[partition]

        self._decay_cluster_traces(cluster)
        self._update_cluster_trace_sum_cache(cluster)

    def clean_up(self):
        # CHANGES CLUSTERS ORDER!!!

        # cluster with the lowest trace
        i = np.argmin(self._clusters_trace)

        # replace i-th with the last and pop new last
        self._clusters[i] = self._clusters[-1]
        self._traces[i] = self._traces[-1]
        self._sizes[i] = self._sizes[-1]
        self._clusters_trace[i] = self._clusters_trace[-1]
        self._cache_traces_sum[i] = self._cache_traces_sum[-1]

        self._traces[-1, :] = 0
        self._sizes[-1] = 0
        self._clusters_trace[-1] = 0
        self._cache_traces_sum[-1] = 0
        self.n_clusters -= 1

    def match(self, sdr: SparseSdr, similarity: np.ndarray = None):
        if self.empty:
            return None, None, np.array([])

        if similarity is None:
            similarity = self.similarity(sdr)

        ind = np.argmax(similarity)
        if similarity[ind] < self.similarity_threshold:
            return None, None, similarity
        return self.active_clusters[ind], ind, similarity

    def similarity(
            self, sdr: SparseSdr, with_active_cluster_parts: bool = False
    ) -> np.ndarray:
        if self.empty:
            return np.array([])

        self._cache_sdr[sdr] = 1
        if with_active_cluster_parts:
            intersection = self._cache_sdr[self.active_clusters]
            trace = self.active_clusters_trace
            trace_sum = self._cache_traces_sum[:self.n_clusters, 1]
        else:
            # non-tracked cluster parts WILL HAVE zero trace => they don't interfere
            intersection = self._cache_sdr[self.clusters]
            trace = self.traces
            trace_sum = self._cache_traces_sum[:self.n_clusters, 0]

        similarity = np.sum(intersection * trace, axis=-1) / trace_sum

        # zeroes back to full zero
        self._cache_sdr[sdr] = 0
        return similarity

    def overlap(self, sdr: SparseSdr) -> np.ndarray:
        if self.empty:
            return np.array([])

        self._cache_sdr[sdr] = 1
        overlap = self._cache_sdr[self.active_clusters].sum(axis=1)
        # zeroes back to full zero
        self._cache_sdr[sdr] = 0
        return overlap

    def overlap2(self, sdr: SparseSdr) -> np.ndarray:
        if self.empty:
            return np.array([])

        # noinspection PyUnresolvedReferences
        overlap = np.ndarray([
            np.intersect1d(sdr, cluster, assume_unique=True).shape[0]
            for cluster in self.active_clusters
        ])
        return overlap

    def change_threshold(self, delta=None):
        if 0 < self.similarity_threshold < self.similarity_threshold_limit:
            self.similarity_threshold += isnone(delta, self.similarity_threshold_delta)

    def _decay_cluster_traces(self, active_cluster: int):
        self._clusters_trace[:self.n_clusters] *= .99
        self._clusters_trace[active_cluster] += 1.

    def _update_cluster_trace_sum_cache(self, cluster: int):
        self._cache_traces_sum[cluster, 0] = np.sum(self.traces[cluster])
        self._cache_traces_sum[cluster, 1] = np.sum(self.active_clusters_trace[cluster])
