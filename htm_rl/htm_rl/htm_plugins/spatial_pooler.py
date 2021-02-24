from typing import Tuple

import numpy as np
from htm.bindings.algorithms import SpatialPooler as HtmSpatialPooler
from htm.bindings.sdr import SDR


class SpatialPooler:
    output_sdr_size: int

    _spatial_pooler: HtmSpatialPooler
    _cached_input_sdr: SDR
    _cached_output_sdr: SDR

    def __init__(
            self, potential_synapses_ratio: float,
            sparsity: float, synapse_permanence_deltas: Tuple[float, float],
            connected_permanence_threshold: float, boost_strength: float,
            boost_sliding_window: int, expected_normal_overlap_frequency: float,
            seed: int, min_activation_threshold: int = 1,
            input_size: int = None, input_source=None,
            output_size: int = None, output_dilation_ratio: float = None,
    ):
        if input_size is None:
            input_size = input_source.output_sdr_size
        input_shape = [input_size]

        if output_size is None:
            output_size = int(output_dilation_ratio * input_size)
        output_shape = [output_size]
        self.output_sdr_size = output_size

        permanence_increase, permanence_decrease = synapse_permanence_deltas
        self._spatial_pooler = HtmSpatialPooler(
            inputDimensions=input_shape,
            columnDimensions=output_shape,
            potentialRadius=input_size,
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

        self._cached_input_sdr = SDR(input_shape)
        self._cached_output_sdr = SDR(output_shape)

    def compute(self, sparse_sdr, learn: bool = True):
        self._cached_input_sdr.sparse = sparse_sdr
        self._spatial_pooler.compute(
            self._cached_input_sdr,
            learn=learn,
            output=self._cached_output_sdr
        )
        return np.array(self._cached_output_sdr.sparse, copy=True)
