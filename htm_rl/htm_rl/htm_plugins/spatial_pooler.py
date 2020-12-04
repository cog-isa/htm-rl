from typing import Tuple

from htm.bindings.algorithms import SpatialPooler as SP


class SpatialPooler(SP):
    def __init__(
            self, input_size: int, output_size: int, permanence_threshold: float,
            sparsity: float, synapse_permanence_deltas: Tuple[float, float],
            connected_permanence_threshold: float, boost_strength: float,
            boost_sliding_window: int, expected_normal_overlap_frequency: float,
            seed: int, min_activation_threshold: int = 1
    ):
        permanence_increase, permanence_decrease = synapse_permanence_deltas

        super(SpatialPooler, self).__init__(
            inputDimensions=[input_size], columnDimensions=[output_size],
            globalInhibition=True,
            potentialPct=permanence_threshold,
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