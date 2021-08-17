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


config = dict(
    activeOverlapWeight=1.0,
    predictedActiveOverlapWeight=0.0,
    maxUnionActivity=0.20,
    exciteFunctionType='Fixed',
    decayFunctionType='NoDecay',
    decayTimeConst=20.0,
    synPermPredActiveInc=0.0,
    synPermPreviousPredActiveInc=0.0,
    historyLength=0,
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

    for x in sequence:
        sdr.dense = x
        tp.compute(sdr, sdr, True)
        visualize(tp, 25, -1, step)


if __name__ == '__main__':
    main()
