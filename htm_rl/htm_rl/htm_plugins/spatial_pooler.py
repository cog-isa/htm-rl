from typing import Tuple

from htm.bindings.algorithms import SpatialPooler as SP
from htm.bindings.sdr import SDR


class SpatialPooler:
    spatial_pooler: SP
    _cached_input_sdr: SDR
    _cached_output_sdr: SDR

    def __init__(
            self, input_size: int, output_size: int, potential_synapses_ratio: float,
            sparsity: float, synapse_permanence_deltas: Tuple[float, float],
            connected_permanence_threshold: float, boost_strength: float,
            boost_sliding_window: int, expected_normal_overlap_frequency: float,
            seed: int, min_activation_threshold: int = 1
    ):
        permanence_increase, permanence_decrease = synapse_permanence_deltas
        self.spatial_pooler = SP(
            inputDimensions=[input_size],
            columnDimensions=[output_size],
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
        self._cached_input_sdr = SDR(input_size)
        self._cached_output_sdr = SDR(output_size)

    def encode(self, sdr, learn: bool = True):
        if isinstance(sdr, list):
            self._cached_input_sdr.sparse = sdr
        else:
            self._cached_input_sdr.dense = sdr

        self.spatial_pooler.compute(
            self._cached_input_sdr, learn=learn, output=self._cached_output_sdr
        )
        return list(self._cached_output_sdr.sparse)
