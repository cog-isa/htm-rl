from abc import abstractmethod, ABC

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
