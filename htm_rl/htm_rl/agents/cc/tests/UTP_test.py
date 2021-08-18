from htm.bindings.sdr import SDR
from htm_rl.agents.cc.cortical_column import UnionTemporalPooler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize(tp, h, w, step):
    sdr = tp.getUnionSDR()
    fig = plt.figure()
    fig.suptitle(f'step {step}')
    sns.heatmap(sdr.dense.reshape((h, w)), vmin=0, vmax=1, cbar=False, linewidths=.5)
    plt.show()


config = dict(
    output_sparsity=0.02,
    n_cortical_columns=1,
    cells_per_cortical_column=1000,
    current_cc_id=0,
    activeOverlapWeight=0.5,
    predictedActiveOverlapWeight=0.5,
    maxUnionActivity=0.2,
    exciteFunctionType='Fixed',
    decayFunctionType='NoDecay',
    decayTimeConst=20.0,
    prune_zero_synapses_basal=True,
    activation_threshold_basal=18,
    learning_threshold_basal=15,
    connected_threshold_basal=0.5,
    initial_permanence_basal=0.4,
    permanence_increment_basal=0.1,
    permanence_decrement_basal=0.01,
    sample_size_basal=20,
    max_synapses_per_segment_basal=20,
    max_segments_per_cell_basal=32,
    timeseries=True,
    synPermPredActiveInc=0.1,
    synPermPreviousPredActiveInc=0.05,
    historyLength=5,
    minHistory=0,
    boostStrength=0.0,
    columnDimensions=[1000],
    inputDimensions=[10],
    potentialRadius=1000,
    dutyCyclePeriod=1000,
    globalInhibition=True,
    localAreaDensity=0.04,
    minPctOverlapDutyCycle=0.001,
    numActiveColumnsPerInhArea=0,
    potentialPct=0.5,
    seed=0,
    spVerbosity=0,
    stimulusThreshold=1,
    synPermActiveInc=0.1,
    synPermConnected=0.1,
    synPermInactiveDec=0.01,
    wrapAround=True
)


def main():
    tp = UnionTemporalPooler(**config)

    sequence = [(np.random.rand(config['inputDimensions'][0]) > 0.5).astype('uint32') for _ in range(10)]

    step = 0
    sdr = SDR(config['inputDimensions'][0])
    prev_out = SDR(config['columnDimensions'][0])

    for x in sequence:
        sdr.dense = x
        tp.compute(sdr, sdr, True, prev_out)
        visualize(tp, 25, -1, step)

        prev_out = tp.getUnionSDR()


if __name__ == '__main__':
    main()
