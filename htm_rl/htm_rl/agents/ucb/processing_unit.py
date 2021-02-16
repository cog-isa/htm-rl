from abc import abstractmethod, ABC
from typing import Sequence, Any

from htm_rl.htm_plugins.spatial_pooler import SpatialPooler


class ProcessingUnit(ABC):
    @property
    @abstractmethod
    def input_shape(self):
        ...

    @property
    @abstractmethod
    def output_shape(self):
        ...

    @abstractmethod
    def process(self, x):
        ...


class ConcatenateUnit(ProcessingUnit):
    _input_shape: Sequence[Any]
    _output_shape: Any

    def __init__(self, input_sources: Sequence[ProcessingUnit]):
        self._input_shape = [
            input_source.input_shape
            for input_source in input_sources
        ]
        self._output_shape = sum(self._input_shape)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def process(self, xs):
        return sum(xs)


class SpatialPoolerUnitTemp(ProcessingUnit):
    spatial_pooler: SpatialPooler

    def __init__(self, input_source: ProcessingUnit, **spatial_pooler_kwargs):
        input_shape = input_source.output_shape
        self.spatial_pooler = SpatialPooler(
            input_size=input_shape,
            **spatial_pooler_kwargs
        )

    @property
    def input_shape(self):
        return self.spatial_pooler.input_shape

    @property
    def output_shape(self):
        return self.spatial_pooler.output_shape

    def process(self, x, learn=True):
        return self.spatial_pooler.encode(x, learn=learn)
