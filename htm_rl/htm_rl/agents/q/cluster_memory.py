import numpy as np

from htm_rl.agents.q.balancing_param import BalancingParam
from htm_rl.agents.q.cluster_memory_stats import ClusterMemoryStats
from htm_rl.common.sdr import SparseSdr, DenseSdr
from htm_rl.common.utils import isnone


class ClusterMemory:
    """
    Organizes input SDRs into clusters. Each cluster is represented as an
    activation density - the more an element is being activated in a particular
    cluster, the more the element's density.
    A cluster can be split into two parts: representatives and the others.
    Representative are the top N most frequently used elements, where
    N = size(SDR) * sparsity is a number of active bits in SDR.

    Any input SDR vector can be matched against all clusters. Match happens based
    on similarity between SDR vector and a cluster. Similarity is computed as
    scalar product between SDR and a cluster's density. Given that clusters
    have unit mass, density may be interpreted as probability, and the similarity
    may be interpreted as an expectation of an SDR vector belonging to a cluster.
    Match happens when similarity is over configurable `similarity_threshold`.
    Similarity threshold can be changed during the course of learning by calling
    `change_threshold` - by default it adds `similarity_threshold_delta` until
    it's over the `similarity_threshold_limit`.

    If an input SDR vector hasn't matched any cluster, it's used to form a new
    cluster. If match happened, the SDR vector is used to update the matched cluster
    density.

    Clusters are kept in a sparse form in `_clusters` as indices of tracked SDR
    elements. A number of elements to track is based on the expected
    number of SDR active bits `n_active_bits` and is configurable by
    `max_tracked_bits_rate` parameter (e.g. =2. means tracking twice the
    number of active bits). Tracked elements with the least density may be
    superseded by the elements from the input SDR that haven't being tracked yet.

    Cluster memory also has limited capacity of clusters defined by the
    configurable param `max_n_clusters`. Adding new cluster to the full memory
    results to the least used cluster being removed in order to free space. Cluster
    usage is tracked with `cluster_trace` as an exponentially decaying trace. The
    decaying rate is not configurable at the moment. BE CAREFUL, removing cluster
    changes the order of clusters, but you can track down this change if needed.
    """
    density_decay: float = .9

    sdr_size: int
    n_active_bits: int
    similarity_threshold: BalancingParam

    stats: ClusterMemoryStats

    n_clusters: int

    # pre-allocated buffer: cluster [tracked elements] indices
    # shape: (max_n_clusters, max_n_tracked_bits)
    _clusters: np.ndarray
    _clusters_sets: list[set[int]]

    # pre-allocated buffer: cluster [tracked elements] density
    # shape: (max_n_clusters, max_n_tracked_bits)
    _density: np.ndarray

    # pre-allocated buffer: how many elements are tracked for each cluster
    # shape: (max_n_clusters)
    _cluster_sizes: np.ndarray
    # pre-allocated buffer: traces of cluster activity
    # shape: (max_n_clusters)
    _cluster_traces: np.ndarray

    # pre-allocated cache vector for sparse-to-dense input SDR vector conversion
    # NOTE: it should always be zeroed outside of any method call.
    _cache_sdr: DenseSdr

    def __init__(
            self, input_sdr_size: int, n_active_bits: int,
            similarity_threshold: dict,
            max_n_clusters: int, max_tracked_bits_rate: float = 3.,
    ):
        self.sdr_size = input_sdr_size
        self.n_active_bits = n_active_bits
        self.similarity_threshold = BalancingParam(**similarity_threshold)

        max_n_tracked_bits = int(n_active_bits * max_tracked_bits_rate)
        self.n_clusters = 0
        self._clusters = np.zeros((max_n_clusters, max_n_tracked_bits), dtype=np.int)
        self._clusters_sets = []
        self._density = np.zeros_like(self._clusters, dtype=np.float)

        self._cluster_sizes = np.zeros(max_n_clusters, dtype=np.int)
        self._cluster_traces = np.zeros(max_n_clusters, dtype=np.float)

        self._cache_sdr = np.zeros(input_sdr_size, dtype=np.int)
        self.stats = ClusterMemoryStats()

    @property
    def empty(self):
        return self.n_clusters == 0

    @property
    def full(self):
        return self.n_clusters == self._clusters.shape[0]

    @property
    def clusters(self):
        """Gets indices of clusters' tracked elements."""
        return self._clusters[:self.n_clusters]

    @property
    def density(self):
        """Gets density of clusters' tracked elements."""
        return self._density[:self.n_clusters]

    @property
    def representatives(self):
        """Gets indices of clusters' representatives."""
        return self.clusters[:, -self.n_active_bits:]

    @property
    def representatives_density(self):
        """Gets density of clusters' representatives."""
        return self.density[:, -self.n_active_bits:]

    def activate(
            self, sdr: SparseSdr,
            similarity: np.ndarray = None, matched_cluster: int = None
    ) -> int:
        """
        Takes into account new activation - an input SDR vector - by adding
        it as a new cluster or by updating density of the best matched cluster.

        Parameters
        ----------
        sdr : SparseSdr
            An input sparse vector.
        similarity : np.ndarray, optional
            Pre-computed similarity for provided input vector.
        matched_cluster : int, optional
            Pre-computed best matched cluster index. If passed, `similarity`
            should be passed too.

        Returns
        ----------
        int
            Matched cluster index.
        """
        if similarity is None:
            cluster, i_cluster, similarity = self.match(sdr)
            matched_cluster = i_cluster

        if matched_cluster is None:
            matched_cluster = self.create(sdr)
        else:
            self.update(matched_cluster, sdr)

        return matched_cluster

    def match(
            self, sdr: SparseSdr,
            similarity: np.ndarray = None, similarity_threshold: float = None
    ):
        """
        Finds the best matched cluster for provided vector `sdr`.

        Parameters
        ----------
        sdr : SparseSdr
            Vector that should be matched.
        similarity : np.ndarray, optional
            Pre-computed similarity between clusters and provided input vector.
        similarity_threshold : float, optional
            Overrides ClusterMemory object's similarity threshold for this call.

        Returns
        -------
        SparseSdr or None
            Best matched cluster. None, if no match.
        int or None
            Best matched cluster index. None, if no match.
        np.ndarray
            Similarity between clusters and provided input vector.
        """
        if self.empty:
            return None, None, np.array([])

        if similarity is None:
            similarity = self.similarity(sdr)

        ind = np.argmax(similarity)
        similarity_threshold = isnone(
            similarity_threshold, self.similarity_threshold.value
        )

        # log stats
        self.stats.on_match(
            self.n_clusters, similarity[ind], similarity_threshold,
            self._cluster_traces[ind]
        )

        if similarity[ind] < similarity_threshold:
            return None, None, similarity
        return self.representatives[ind], ind, similarity

    def similarity(
            self, sdr: SparseSdr, with_active_cluster_parts: bool = False
    ) -> np.ndarray:
        """
        Computes similarity between an input sparse vector `sdr` and
        all clusters.

        Parameters
        ----------
        sdr : SparseSdr
            An input sparse vector.
        with_active_cluster_parts : bool, optional
            Specifies, whether to compute similarity against all tracked cluster
            elements or only against cluster representatives. False, by default.

        Returns
        -------
        np.ndarray
            Similarity between clusters and provided input vector.
        """
        if self.empty:
            return np.array([])

        # sparse to dense `sdr` using zeroed cache
        self._cache_sdr[sdr] = 1

        if with_active_cluster_parts:
            # which representatives are activated by `sdr`
            clusters_activation_mask = self._cache_sdr[self.representatives]
            density = self.representatives_density
        else:
            # which clusters' tracked elements are activated by `sdr`
            clusters_activation_mask = self._cache_sdr[self.clusters]
            # non-tracked cluster elements ARE GUARANTEED to have zero density
            density = self.density

        # _row-wise_ scalar product
        similarity = np.sum(clusters_activation_mask * density, axis=-1)

        # zeroes back cache to follow contract
        self._cache_sdr[sdr] = 0
        return similarity

    def create(self, sdr: SparseSdr) -> int:
        """
        Creates new cluster from provided sparse vector. If cluster memory is full,
        the least used cluster will be removed to free up space.

        Parameters
        ----------
        sdr : SparseSdr
            An input sparse vector that will define new cluster representatives.

        Returns
        -------
        int
            Index of the added cluster.
        """
        if self.full:
            self.remove_least_used_cluster()

        i_cluster = self.n_clusters
        n = len(sdr)

        self._clusters[i_cluster, -n:] = sdr
        # noinspection PyTypeChecker
        self._clusters_sets.append(set(sdr.tolist()))
        # all clusters have unit mass
        self._density[i_cluster, -n:] = 1. / n
        self._cluster_sizes[i_cluster] = n
        self.n_clusters += 1
        self._update_cluster_traces(i_cluster)

        # log stats
        self.stats.on_added()

        return i_cluster

    def update(self, cluster: int, sdr: SparseSdr):
        """
        Updates selected cluster's density corresponding to provided activation.

        Parameters
        ----------
        cluster : int
            Index of an updating cluster.
        sdr : SparseSdr
            Activation that updates cluster's density.
        """
        size = self._cluster_sizes[cluster]
        max_cluster_size = self._clusters.shape[1]

        # noinspection PyTypeChecker
        sdr_set = set(sdr.tolist())
        # get elements that should be added to cluster
        to_add = sdr_set - self._clusters_sets[cluster]
        k = len(to_add)
        mass = 1.

        # if not enough space to add new elements, remove elements with the
        # least density to get needed space
        if max_cluster_size < size + k:
            # Re-partition to place k elements with the lowest density on the left.
            # Non-tracked elements are guaranteed to be here too
            # as their density is 0.
            partition = np.argpartition(self._density[cluster], k, axis=-1)
            self._clusters[cluster, :] = self._clusters[cluster, partition]
            self._density[cluster, :] = self._density[cluster, partition]

            cleared = self._clusters[cluster, :k]
            cleared_density = self._density[cluster, :k]
            # get removed tracked elements
            cleared_set = set(cleared[cleared_density > 1e-10].tolist())
            # have to remove from the initial sdr those that a removed
            sdr_set -= cleared_set
            self._clusters_sets[cluster] -= cleared_set

            size = max_cluster_size - k
            # exclude removed elements' density from cluster mass
            mass -= cleared_density.sum()

        if k > 0:
            # append new elements
            self._clusters[cluster, -size-k:-size] = list(to_add)
            self._density[cluster, -size-k:-size] = 0.
            size += k

            # update cluster set and size
            self._clusters_sets[cluster] |= to_add
            self._cluster_sizes[cluster] = size

        # update density: a) decay for tracked
        self._density[cluster, -size:] *= self.density_decay
        mass *= self.density_decay

        # b) uniformly distribute mass among activated elements
        # to restore cluster's unit mass
        # NOTE: it's crucial that the removed elements are removed from sdr_set too
        delta_density = (1. - mass) / len(sdr_set)
        activated_elem_indices = [
            -i
            for i in range(1, size+1)
            if self._clusters[cluster, -i] in sdr_set
        ]
        self._density[cluster, activated_elem_indices] += delta_density

        # repartition cluster to move representatives to the right side
        valid_cluster_part = self._clusters[cluster, -size:]
        valid_density_part = self._density[cluster, -size:]
        repr_size = min(self.n_active_bits, size)
        partition = np.argpartition(valid_density_part, -repr_size, axis=-1)
        self._density[cluster, -size:] = valid_density_part[partition]
        self._clusters[cluster, -size:] = valid_cluster_part[partition]

        # update cluster trace
        self._update_cluster_traces(cluster)

    def remove_least_used_cluster(self) -> int:
        """
        Removes the least used cluster. NOTE: CHANGES CLUSTERS ORDER!!!

        Returns
        -------
        int
            Index of the removed cluster. It could be used to get the cluster
            ordering change - selected for removal cluster is swapped with the
            last [in clusters buffer] cluster and then is popped from buffer.
            Hence, after removal we have: a) i-th cluster removed, b) -1-th
            cluster moved to i-th position.
        """
        # cluster with the lowest trace
        i = np.argmin(self._cluster_traces)

        removed_cluster = self.representatives[i].copy()
        removed_cluster_trace = self._cluster_traces[i]

        # replace i-th with the last and pop last
        self._clusters[i] = self._clusters[-1]
        self._clusters_sets[i] = self._clusters_sets[-1]
        self._density[i] = self._density[-1]
        self._cluster_sizes[i] = self._cluster_sizes[-1]
        self._cluster_traces[i] = self._cluster_traces[-1]

        # no need to clean up _clusters
        self._density[-1, :] = 0
        self._clusters_sets.pop()

        self._cluster_sizes[-1] = 0
        self._cluster_traces[-1] = 0
        self.n_clusters -= 1

        # log stats
        best_match_similarity = np.max(self.similarity(removed_cluster))
        self.stats.on_removed(removed_cluster_trace, best_match_similarity)

        # noinspection PyTypeChecker
        return i

    def overlap(self, sdr: SparseSdr, with_active_cluster_parts=False) -> np.ndarray:
        if self.empty:
            return np.array([])

        if with_active_cluster_parts:
            # noinspection PyUnresolvedReferences
            overlap = np.ndarray([
                np.intersect1d(sdr, cluster, assume_unique=True).shape[0]
                for cluster in self.representatives
            ])
        else:
            self._cache_sdr[sdr] = 1
            overlap = self._cache_sdr[self.representatives].sum(axis=1)
            # zeroes back to full zero
            self._cache_sdr[sdr] = 0
        return overlap

    def _update_cluster_traces(self, active_cluster: int):
        self._cluster_traces[:self.n_clusters] *= .99
        self._cluster_traces[active_cluster] += 1.
