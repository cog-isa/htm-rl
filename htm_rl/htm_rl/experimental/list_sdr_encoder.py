# from typing import List, NamedTuple, Iterable
#
# from htm_rl.minigrid.sar import SuperpositionList
# from htm_rl.common.int_sdr_encoder import IntSdrEncoder, BitRange
# from htm_rl.common.sdr import SparseSdr
# from htm_rl.common.utils import isnone
#
# Dim2d = NamedTuple('Dim2d', (('rows', int), ('cols', int)))
#
#
# """
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Outdated and probably doesn't work at all.
# Kept for future needs. Will be adapted or deleted when the time is come.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# """
#
#
# class ListSdrEncoder:
#     value_bits: int
#     total_bits: int
#     activation_threshold: int
#
#     _elem_encoder: IntSdrEncoder
#     _n_elems: int
#     _n_dim: Dim2d
#     _shifts: List[int]
#
#     def __init__(self, elem_encoder: IntSdrEncoder, n_dim: Dim2d):
#         n_rows, n_cols = n_dim
#         n_elems = n_rows * n_cols
#
#         self.value_bits = n_elems * elem_encoder.value_bits
#         self.total_bits = n_elems * elem_encoder.total_bits
#         self.activation_threshold = n_elems * elem_encoder.activation_threshold
#
#         self._elem_encoder = elem_encoder
#         self._n_elems = n_elems
#         self._n_dim = n_dim
#         self._shifts = self._get_shifts(elem_encoder, n_elems)
#
#     def encode(self, values) -> Iterable[BitRange]:
#         values = isnone(values, [])
#         return (
#             BitRange(bit_range.l + shift, bit_range.r + shift)
#             for x, shift in zip(values, self._shifts)
#             for bit_range in self._elem_encoder.encode(x)
#         )
#
#     def decode(self, indices: SparseSdr) -> SuperpositionList:
#         elem_sdrs = self.split_to_elems(indices)
#         return list(map(self._elem_encoder.decode, elem_sdrs))
#
#     def split_to_elems(self, indices: SparseSdr) -> List[SparseSdr]:
#         elem_buckets = [[] for _ in range(self._n_elems)]
#         for ind in indices:
#             elem, intra_elem_ind = divmod(ind, self._elem_encoder.total_bits)
#             elem_buckets[elem].append(intra_elem_ind)
#         return elem_buckets
#
#     @staticmethod
#     def _get_shifts(encoder: IntSdrEncoder, n_elems: int) -> List[int]:
#         step = encoder.total_bits
#         return list(range(0, step * n_elems, step))
#
#
# class ListSdrFormatter:
#     encoder: ListSdrEncoder
#     rows: int
#     cols: int
#
#     def __init__(self, encoder: ListSdrEncoder, elem_formatter, rows: int, cols: int):
#         self.encoder = encoder
#         self.elem_formatter = elem_formatter
#         self.rows = rows
#         self.cols = cols
#
#     def format(self, indices: SparseSdr) -> str:
#         elem_sdrs = self.encoder.split_to_elems(indices)
#         rows, cols = self.rows, self.cols
#
#         return '\n'.join(
#             ' | '.join(
#                 self.elem_formatter.format_sar_superposition(elem_sdrs[row * cols + col])
#                 for col in range(cols)
#             )
#             for row in range(rows)
#         )
