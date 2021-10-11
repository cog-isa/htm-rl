from htm_rl.common.utils import safe_divide


class ClusterMemoryStats:
    clusters: int
    added: int
    matched: int
    mismatched: int
    removed: int

    sum_similarity_threshold: float
    sum_match_similarity: float
    sum_match_cluster_trace: float
    sum_mismatch_similarity: float
    sum_removed_best_match_similarity: float
    sum_removed_cluster_trace: float

    def __init__(self):
        self.clusters = 0
        self.reset()

    def reset(self):
        self.added = 0
        self.matched = 0
        self.mismatched = 0
        self.removed = 0
        self.sum_match_similarity = 0.
        self.sum_mismatch_similarity = 0.
        self.sum_similarity_threshold = 0.
        self.sum_match_cluster_trace = 0.
        self.sum_removed_best_match_similarity = 0.
        self.sum_removed_cluster_trace = 0.

    @property
    def avg_match_rate(self):
        return safe_divide(self.matched, self.matched + self.mismatched)

    @property
    def avg_match_similarity(self):
        return safe_divide(self.sum_match_similarity, self.matched)

    @property
    def avg_mismatch_similarity(self):
        return safe_divide(self.sum_mismatch_similarity, self.mismatched)

    @property
    def avg_matched_cluster_trace(self):
        return safe_divide(self.sum_match_cluster_trace, self.matched)

    @property
    def avg_similarity_threshold(self):
        return safe_divide(self.sum_similarity_threshold, self.matched + self.mismatched)

    @property
    def avg_removed_cluster_intra_similarity(self):
        return safe_divide(self.sum_removed_best_match_similarity, self.removed)

    @property
    def avg_removed_cluster_trace(self):
        return safe_divide(self.sum_removed_cluster_trace, self.removed)

    def on_match(
            self, clusters: int, similarity: float, threshold: float,
            best_match_cluster_trace: float
    ):
        self.clusters = clusters
        self.sum_similarity_threshold += threshold
        if similarity < threshold:
            # mismatch
            self.mismatched += 1
            self.sum_mismatch_similarity += similarity
        else:
            # match
            self.matched += 1
            self.sum_match_similarity += similarity
            self.sum_match_cluster_trace += best_match_cluster_trace

    def on_added(self):
        self.added += 1

    def on_removed(self, removed_cluster_trace: float, best_match_similarity: float):
        self.removed += 1
        self.sum_removed_cluster_trace += removed_cluster_trace
        self.sum_removed_best_match_similarity += best_match_similarity
