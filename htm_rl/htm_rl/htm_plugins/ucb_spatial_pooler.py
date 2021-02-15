from typing import Tuple, List

from htm.bindings.algorithms import SpatialPooler as SP
from htm.bindings.sdr import SDR

from htm_rl.agents.ucb.processing_unit import ProcessingUnit


class UcbSpatialPooler(ProcessingUnit):
    spatial_pooler: SP
    _input_shape: List[int]
    _output_shape: List[int]
    _cached_input_sdr: SDR
    _cached_output_sdr: SDR

    def __init__(
            self, output_size: int, potential_synapses_ratio: float,
            sparsity: float, synapse_permanence_deltas: Tuple[float, float],
            connected_permanence_threshold: float, boost_strength: float,
            boost_sliding_window: int, expected_normal_overlap_frequency: float,
            seed: int, min_activation_threshold: int = 1,
            input_size: int = None, input_source: ProcessingUnit = None
    ):
        permanence_increase, permanence_decrease = synapse_permanence_deltas
        if input_size is None:
            input_shape = list(input_source.output_shape)
        else:
            input_shape = [input_size]
        output_shape = [output_size]

        self.spatial_pooler = SP(
            inputDimensions=input_shape,
            columnDimensions=output_shape,
            potentialRadius=input_shape[0],
            potentialPct=potential_synapses_ratio,
            globalInhibition=True,
            localAreaDensity=sparsity,
            # min overlapping required to activate output col
            stimulusThreshold=min_activation_threshold,
            synPermConnected=connected_permanence_threshold,
            synPermActiveInc=permanence_increase,
            synPermInactiveDec=permanence_decrease,
            boostStrength=boost_strength,
            dutyCyclePeriod=boost_sliding_window,
            minPctOverlapDutyCycle=expected_normal_overlap_frequency,
            seed=seed,
        )

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._cached_input_sdr = SDR(input_shape)
        self._cached_output_sdr = SDR(output_shape)

    def encode(self, sdr, learn: bool = True):
        if isinstance(sdr, list):
            self._cached_input_sdr.sparse = sdr
        else:
            self._cached_input_sdr.dense = sdr

        self.spatial_pooler.compute(
            self._cached_input_sdr, learn=learn, output=self._cached_output_sdr
        )
        return list(self._cached_output_sdr.sparse)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def process(self, x, learn=True):
        return self.encode(x, learn=learn)
