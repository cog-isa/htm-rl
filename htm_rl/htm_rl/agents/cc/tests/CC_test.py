#!/usr/bin/env python
# coding: utf-8

# In[1]:


from htm_rl.agents.cc.cortical_column import GeneralFeedbackTM

# In[2]:


from htm.bindings.sdr import SDR
from htm_rl.agents.cc.cortical_column import UnionTemporalPooler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# In[3]:


from htm_rl.common.sdr_encoders import IntBucketEncoder


# In[16]:


def visualize_tp(tp, h, w, step=0):
    sdr = tp.getUnionSDR()
    fig = plt.figure()
    fig.suptitle(f'step {step}')
    sns.heatmap(sdr.dense.reshape((h, w)), vmin=0, vmax=1, cbar=False, linewidths=.5)


# In[15]:


def visualize_tm(tm: GeneralFeedbackTM, feedback_shape=(1, -1), step: int = 0):
    fig, ax = plt.subplots(nrows=3, ncols=2)

    sns.heatmap(tm.active_cells_feedback.dense[tm.feedback_range[0]:tm.feedback_range[1]].reshape(feedback_shape),
                ax=ax[0, 1], linewidths=.5, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1.5)
    sns.heatmap(tm.active_cells_context.dense[tm.context_range[0]:tm.context_range[1]][None],
                ax=ax[1, 0], linewidths=.5, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1.5)
    sns.heatmap((tm.active_cells.dense[tm.local_range[0]:tm.local_range[1]].reshape((tm.columns, -1)).T +
                 tm.predicted_cells.dense[tm.local_range[0]:tm.local_range[1]].reshape((tm.columns, -1)).T * 0.5),
                ax=ax[1, 1], linewidths=.5, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1.5)
    sns.heatmap(tm.active_columns.dense[None],
                ax=ax[2, 1], linewidths=.5, cbar=False, xticklabels=False, yticklabels=False, vmin=0, vmax=1.5)

    ax[0, 1].set_title('Feedback')
    ax[1, 0].set_title('Context')
    ax[1, 1].set_title('')
    ax[2, 1].set_title('Input')

    ax[0, 0].axis('off')
    ax[0, 0].text(x=0.5, y=0.5, s=str(tm.anomaly[-1]))
    ax[0, 0].set_title('anomaly')
    ax[2, 0].axis('off')
    ax[2, 0].text(x=0.5, y=0.5, s=str(tm.confidence[-1]))
    ax[2, 0].set_title('confidence')
    fig.tight_layout(pad=1.0)
    fig.suptitle(f'step: {step}')


# In[36]:


n_actions = 4
action_bucket = 10
n_states = 25
state_bucket = 3
action_encoder = IntBucketEncoder(n_actions, action_bucket)
state_encoder = IntBucketEncoder(n_states, state_bucket)

# In[37]:


input_columns = action_encoder.output_sdr_size
input_active_cells = action_bucket
cells_per_column = 16
output_columns = 4000
output_active_cells = 40
seed = 8543
config_tm = dict(columns=input_columns,
                 cells_per_column=cells_per_column,
                 context_cells=state_encoder.output_sdr_size,
                 feedback_cells=output_columns,
                 activation_threshold_basal=state_bucket,
                 learning_threshold_basal=state_bucket,
                 activation_threshold_apical=output_active_cells,
                 learning_threshold_apical=output_active_cells,
                 connected_threshold_basal=0.5,
                 permanence_increment_basal=0.1,
                 permanence_decrement_basal=0.01,
                 initial_permanence_basal=0.4,
                 predicted_segment_decrement_basal=0.001,
                 sample_size_basal=state_bucket,
                 max_synapses_per_segment_basal=state_bucket,
                 max_segments_per_cell_basal=32,
                 connected_threshold_apical=0.5,
                 permanence_increment_apical=0.1,
                 permanence_decrement_apical=0.01,
                 initial_permanence_apical=0.4,
                 predicted_segment_decrement_apical=0.001,
                 sample_size_apical=output_active_cells,
                 max_synapses_per_segment_apical=output_active_cells,
                 max_segments_per_cell_apical=32,
                 prune_zero_synapses=True,
                 timeseries=False,
                 anomaly_window=1000,
                 confidence_window=1000,
                 noise_tolerance=0.0,
                 sm_ac=0.99,
                 seed=seed)
config_tp = dict(
    output_sparsity=0.01,
    n_cortical_columns=1,
    cells_per_cortical_column=output_columns,
    current_cc_id=0,
    activeOverlapWeight=0.5,
    predictedActiveOverlapWeight=0.5,
    maxUnionActivity=0.5,
    exciteFunctionType='Fixed',
    decayFunctionType='NoDecay',
    decayTimeConst=20.0,
    prune_zero_synapses_basal=True,
    activation_threshold_basal=output_active_cells,
    learning_threshold_basal=output_active_cells,
    connected_threshold_basal=0.5,
    initial_permanence_basal=0.4,
    permanence_increment_basal=0.1,
    permanence_decrement_basal=0.01,
    sample_size_basal=output_active_cells,
    max_synapses_per_segment_basal=output_active_cells,
    max_segments_per_cell_basal=32,
    timeseries=True,
    synPermPredActiveInc=0.1,
    synPermPreviousPredActiveInc=0.05,
    historyLength=5,
    minHistory=5,
    boostStrength=1.0,
    columnDimensions=[output_columns],
    inputDimensions=[input_columns*cells_per_column],
    potentialRadius=input_columns*cells_per_column,
    dutyCyclePeriod=1000,
    globalInhibition=True,
    localAreaDensity=0.01,
    minPctOverlapDutyCycle=0.001,
    numActiveColumnsPerInhArea=0,
    potentialPct=0.5,
    seed=seed,
    spVerbosity=0,
    stimulusThreshold=3,
    synPermActiveInc=0.1,
    synPermConnected=0.5,
    synPermInactiveDec=0.01,
    wrapAround=True
)


# In[40]:

def run(tm, tp, policy, learn=True, repeat=1, visualize=False):
    tp_input = SDR(tp.getNumInputs())
    tp_predictive = SDR(tp.getNumInputs())
    for _ in range(repeat):
        for state, action in policy:
            context = state_encoder.encode(state)
            active_input = action_encoder.encode(action)

            tm.set_active_feedback_cells(tp.getUnionSDR().sparse)
            tm.set_active_context_cells(context)

            tm.activate_apical_dendrites(learn)
            tm.activate_basal_dendrites(learn)
            tm.predict_cells()

            tm.set_active_columns(active_input)
            tm.activate_cells(learn)

            if visualize:
                visualize_tm(tm, feedback_shape=(40, -1))

            tp_input.sparse = tm.get_active_cells()
            tp_predictive.sparse = tm.get_correctly_predicted_cells()
            tp.compute(tp_input, tp_predictive, learn, tp.getUnionSDR())

            if visualize:
                visualize_tp(tp, 40, -1)


def train(tm, tp, data, codes, repeat=5):
    for i, p in enumerate(data):
        run(tm, tp, p, repeat=repeat, learn=True)
        codes[i] = tp.getUnionSDR().dense
        tp.reset(boosting=False)
        tm.reset()


def test(tm, tp, data, codes, threshold):
    n_sensations_per_policy = list()
    for i, policy in enumerate(data):
        n_sensations = 0
        for state, action in policy:
            run(tm, tp, [(state, action)], learn=False)
            n_sensations += 1
            overlap = np.dot(codes, tp.getUnionSDR().dense[None].T)
            if np.sum(overlap >= threshold) == 1:
                n_sensations_per_policy.append(n_sensations)
                break
        else:
            n_sensations_per_policy.append(-1)
        tp.reset()
        tm.reset()
    return n_sensations_per_policy


def test_retrieval(tm, tp, data, codes, threshold):
    n_sensations = test(tm, tp, data, codes, threshold)
    print(n_sensations)
    print(np.sum(np.array(n_sensations) != -1) / len(n_sensations))


def test_coding_similarity(data, codes, n_policies):
    raw_similarity = np.zeros((n_policies, n_policies))
    codes_similarity = np.zeros((n_policies, n_policies))
    for i, seq1 in enumerate(data):
        for j, seq2 in enumerate(data):
            raw_similarity[i, j] = 1 - np.count_nonzero(seq1 - seq2)/seq1.size
            codes_similarity[i, j] = np.dot(codes[i], codes[j]) / codes[i].sum()

    sns.heatmap(raw_similarity)
    plt.show()
    sns.heatmap(codes_similarity)
    plt.show()


def main():
    n_policies = 50
    data1 = [np.random.randint(0, n_actions, n_states) for _ in range(n_policies)]
    data = [list(zip(range(n_states), x)) for x in data1]
    codes = np.zeros((n_policies, output_columns), dtype='int8')
    tm = GeneralFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)
    train(tm, tp, data, codes, repeat=10)

    test_retrieval(tm, tp, data, codes, 30)

    test_coding_similarity(data1, codes, n_policies)


if __name__ == '__main__':
    main()
