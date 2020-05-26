from inspect import ismethod

import numpy as np
from htm.bindings.sdr import SDR
#
# from mdp_planner import DataEncoder, DataMultiEncoder, TemporalMemory, HtmAgent
#
#
# class TestDataEncoder:
#     def __init__(self):
#         self.encoder = DataEncoder('-', n_vals=2, value_bits=3, activation_threshold=2)
#
#     def test_encode_dense(self):
#         result = self.encoder.encode_dense(1)
#         expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
#         assert np.array_equal(result, expected)
#
#     def test_encode_sparse(self):
#         arr_sparse = self.encoder.encode_sparse(1)
#         assert arr_sparse == [3, 4, 5]
#
#     def test_str_from_dense(self):
#         arr_dense = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
#         res = self.encoder.str_from_dense(arr_dense)
#         assert res == '000 111'
#
#     def test_decode_dense(self):
#         decoded = self.encoder.decode_dense(np.array([0, 1, 1, 0, 1, 0]))
#         assert decoded == [0]
#
#         decoded = self.encoder.decode_dense(np.array([0, 1, 1, 1, 1, 1]))
#         assert decoded == [0, 1]
#
#     def test_decode_sparse(self):
#         decoded = self.encoder.decode_sparse([1, 2, 4])
#         assert decoded == [0]
#
#         decoded = self.encoder.decode_sparse([1, 2, 3, 4, 5])
#         assert decoded == [0, 1]
#
#     def test_to_str(self):
#         assert str(self.encoder) == 'DataEncoder("-", v2 x b3)'
#
#
# class TestDataMultiEncoder:
#     def __init__(self):
#         self.encoder = DataMultiEncoder((
#             DataEncoder('1', 2, 4),
#             DataEncoder('2', 3, 2)
#         ))
#
#     def test_encode_dense(self):
#         result = self.encoder.encode_dense((0, 2))
#         expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
#         assert np.array_equal(result, expected)
#
#     def test_encode_sparse(self):
#         result = self.encoder.encode_sparse((1, 1))
#         expected = [4, 5, 6, 7, 10, 11]
#         assert result == expected
#
#     def test_str_from_dense(self):
#         test_arr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
#         result = self.encoder.str_from_dense(test_arr)
#         expected = '1111 0000 00 00 11'
#         assert result == expected
#
#
# class TestHtmAgent:
#     def __init__(self):
#         self.encoder = DataMultiEncoder((
#             DataEncoder('1', 2, 4),
#             DataEncoder('2', 3, 2)
#         ))
#         # total_bits == 14
#         self.tm = TemporalMemory(
#             n_columns=self.encoder.total_bits,
#             cells_per_column=2,
#             activation_threshold=5, learning_threshold=3,
#             initial_permanence=.5, connected_permanence=.5
#         )
#         self.agent = HtmAgent(self.tm, self.encoder)
#
#     def test_str_from_cells(self):
#         active_cells = SDR((self.tm.n_columns, self.tm.cells_per_column))
#         active_cells.dense[5, 0] = 1
#         active_cells.dense[9, 1] = 1
#
#         result = self.agent._str_from_cells(active_cells, 'test_name')
#         expected = '''0000 0100 00 00 00 test_name
# 0000 0000 01 00 00'''
#
#         assert result == expected
#
#
# def _test_all(*objects):
#     def test_all_for_obj(obj):
#         for name in dir(obj):
#             attribute = getattr(obj, name)
#             if ismethod(attribute) and name.startswith('test_'):
#                 attribute()
#
#     for obj in objects:
#         test_all_for_obj(obj)
#
#
# def test_all():
#     _test_all(
#         TestDataEncoder(),
#         TestDataMultiEncoder(),
#         TestHtmAgent()
#     )
