from typing import Union, List

from representations.sar import Sar


# TODO: rollback to non-vectorized
def encode_sar_value(sar: Union[Sar, List[Sar]]) -> Union[int, List[int]]:
    def _encode_sar_value(sar: Sar) -> int:
        s, a, r = sar
        return s * 100 + a * 10 + r

    if isinstance(sar, list):
        sars = sar
        return [_encode_sar_value(sar) for sar in sars]
    return _encode_sar_value(sar)


def decode_sar_value(sar_value: Union[int, List[int]]) -> Union[Sar, List[Sar]]:
    def _decode_sar_value(sar_value: int) -> Sar:
        s, ar = divmod(sar_value, 100)
        a, r = divmod(ar, 10)
        return s, a, r

    if isinstance(sar_value, list):
        sar_values = sar_value
        return [_decode_sar_value(sar_value) for sar_value in sar_values]
    return _decode_sar_value(sar_value)
