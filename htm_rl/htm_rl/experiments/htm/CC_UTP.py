from htm_rl.modules.htm.temporal_memory import GeneralFeedbackTM, DelayedFeedbackTM
from htm_rl.modules.htm.spatial_pooler import UnionTemporalPooler
from htm_rl.common.sdr_encoders import IntBucketEncoder

from htm.bindings.sdr import SDR

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from copy import deepcopy


def visualize_tp(tp, h, w, step=0):
    sdr = tp.getUnionSDR()
    fig = plt.figure()
    fig.suptitle(f'step {step}')
    sns.heatmap(sdr.dense.reshape((h, w)), vmin=0, vmax=1, cbar=False, linewidths=.5)


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


def run(tm, tp, policy, state_encoder, action_encoder, propagate=True, learn=True, visualize=False):
    tp_input = SDR(tp.getNumInputs())
    tp_predictive = SDR(tp.getNumInputs())
    tp_prev_union = SDR(tp.getNumColumns())
    difference = list()
    for state, action in policy:
        context = state_encoder.encode(state)
        active_input = action_encoder.encode(action)

        tm.set_active_context_cells(context)

        tm.activate_basal_dendrites(learn)
        tm.predict_cells()

        tm.set_active_columns(active_input)
        tm.activate_cells(learn)

        if visualize:
            visualize_tm(tm, feedback_shape=(40, -1))

        tp_input.sparse = tm.get_active_cells()
        tp_predictive.sparse = tm.get_correctly_predicted_cells()
        tp.compute(tp_input, tp_predictive, learn)

        difference.append(np.setdiff1d(tp.getUnionSDR().sparse, tp_prev_union.sparse).size)
        tp_prev_union.sparse = tp.getUnionSDR().sparse.copy()

        if visualize:
            visualize_tp(tp, 40, -1)
    if propagate:
        tm.set_active_feedback_cells(tp.getUnionSDR().sparse)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

    return sum(difference)/len(difference)


def train(tm, tp, data, state_encoder, action_encoder, diff_threshold, epochs=5, seed=0, log=False):
    codes = list()
    np.random.seed(seed)
    indices = np.arange(len(data))
    propagate = [False] * len(indices)
    for epoch in range(epochs):
        epoch_codes = dict()
        np.random.shuffle(indices)
        differences = dict()
        for i in indices:
            intra_diff = run(tm, tp, data[i], state_encoder, action_encoder, propagate=propagate[i], learn=True)
            epoch_codes[i] = tp.getUnionSDR().sparse.copy()
            if epoch != 0:
                prev_code = codes[epoch-1][i]
                curr_code = epoch_codes[i]
                difference = np.setdiff1d(prev_code, curr_code).size
                if difference < diff_threshold:
                    propagate[i] = True
                else:
                    propagate[i] = False
            else:
                difference = 0
            differences[f'policy_code_diff{i}'] = difference
            differences[f'policy_intra_diff{i}'] = intra_diff
            if log:
                wandb.log({'tm_apical_segments': tm.apical_connections.numSegments(),
                           'tm_basal_segments': tm.basal_connections.numSegments()
                           })
            tp.reset(boosting=False)
            tm.reset()
        codes.append(deepcopy(epoch_codes))
        differences['epoch'] = epoch
        if log:
            wandb.log(differences)
    return codes[-1]


def test_retrieval(tm, data, codes, threshold, state_encoder, action_encoder):
    action_codes = [action_encoder.encode(x) for x in range(action_encoder.n_values)]
    accuracies = list()
    for i, policy in enumerate(data):
        accuracy = 0
        for state, action in policy:
            context = state_encoder.encode(state)

            tm.set_active_feedback_cells(codes[i])
            tm.set_active_context_cells(context)

            tm.activate_apical_dendrites(learn=False)
            tm.activate_basal_dendrites(learn=False)
            tm.predict_cells()

            prediction = tm.get_predicted_columns()
            overlaps = np.zeros(len(action_codes))
            for j, a_code in enumerate(action_codes):
                overlaps[j] = np.intersect1d(prediction, a_code).size
            mask = np.flatnonzero(overlaps >= threshold)
            if mask.size == 1:
                if mask[0] == action:
                    accuracy += 1
        accuracies.append(accuracy/len(policy))
        tm.reset()
    return sum(accuracies)/len(accuracies)


def test_coding_similarity(data, codes, n_policies, log=False):
    raw_similarity = np.zeros((n_policies, n_policies))
    codes_similarity = np.zeros((n_policies, n_policies))
    for i, seq1 in enumerate(data):
        s1 = set(seq1)
        for j, seq2 in enumerate(data):
            s2 = set(seq2)
            raw_similarity[i, j] = len(s1.intersection(s2))/len(s1.union(s2))
            codes_similarity[i, j] = np.intersect1d(codes[i], codes[j]).size / (np.union1d(codes[i], codes[j]).size + 1e-12)

    if log:
        wandb.log({'raw_similarity': wandb.Image(sns.heatmap(raw_similarity, vmin=0, vmax=1)),
                   'codes_similarity': wandb.Image(sns.heatmap(codes_similarity, cbar=False, vmin=0, vmax=1))})
        plt.close('all')
    else:
        sns.heatmap(raw_similarity)
        plt.show()
        sns.heatmap(codes_similarity)
        plt.show()


def generate_data(n, n_actions, n_states, randomness=1.0, seed=0):
    raw_data = list()
    np.random.seed(seed)
    seed_seq = np.random.randint(0, n_actions, n_states)
    raw_data.append(seed_seq.copy())
    n_replace = int(n_states * randomness)
    for i in range(1, n):
        new_seq = np.random.randint(0, n_actions, n_states)
        if randomness == 1.0:
            raw_data.append(new_seq)
        else:
            indices = np.random.randint(0, n_states, n_replace)
            seed_seq[indices] = new_seq[indices]
            raw_data.append(seed_seq.copy())
    data = [list(zip(range(n_states), x)) for x in raw_data]
    return raw_data, data


def run_config(config):
    action_encoder = IntBucketEncoder(config['n_actions'], config['action_bucket'])
    state_encoder = IntBucketEncoder(config['n_states'], config['state_bucket'])
    raw_data, data = generate_data(config['n_policies'], config['n_actions'], config['n_states'], randomness=config['randomness'], seed=config['seed'])
    tm = DelayedFeedbackTM(**config['tm'])
    tp = UnionTemporalPooler(**config['tp'])
    codes = train(tm, tp, data, state_encoder, action_encoder, diff_threshold=int(
                         config['output_union_sparsity'] * config['output_columns'] * config['noise_tolerance_apical']), epochs=config['epochs'], log=config['log_train'])

    retrieval_accuracy = test_retrieval(tm, data, codes, config['threshold'], state_encoder, action_encoder)
    test_coding_similarity(data, codes, config['n_policies'], log=config['log_codes'])
    return retrieval_accuracy


def main(seeds: list):
    n_policies = 5
    epochs = 50
    threshold = 8
    randomness = 1.0

    n_actions = 4
    action_bucket = 10
    n_states = 25
    state_bucket = 3
    action_encoder = IntBucketEncoder(n_actions, action_bucket)
    state_encoder = IntBucketEncoder(n_states, state_bucket)

    input_columns = action_encoder.output_sdr_size
    cells_per_column = 16
    output_columns = 4000
    output_union_sparsity = 0.01
    noise_tolerance_apical = 0.1
    learning_margin_apical = 0.2
    config_tm = dict(columns=input_columns,
                     cells_per_column=cells_per_column,
                     context_cells=state_encoder.output_sdr_size,
                     feedback_cells=output_columns,
                     activation_threshold_basal=state_bucket,
                     learning_threshold_basal=state_bucket,
                     activation_threshold_apical=int(
                         output_union_sparsity * output_columns * (1 - noise_tolerance_apical)),
                     learning_threshold_apical=int(
                         output_union_sparsity * output_columns * (1 - learning_margin_apical)),
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
                     sample_size_apical=int(output_union_sparsity * output_columns),
                     max_synapses_per_segment_apical=int(output_union_sparsity * output_columns),
                     max_segments_per_cell_apical=32,
                     prune_zero_synapses=True,
                     timeseries=False,
                     anomaly_window=1000,
                     confidence_window=1000,
                     noise_tolerance=0.0,
                     sm_ac=0.99)
    config_tp = dict(
        activeOverlapWeight=1,
        predictedActiveOverlapWeight=2,
        maxUnionActivity=output_union_sparsity,
        exciteFunctionType='Logistic',
        decayFunctionType='Exponential',
        decayTimeConst=10.0,
        synPermPredActiveInc=0.1,
        synPermPreviousPredActiveInc=0.05,
        historyLength=20,
        minHistory=3,
        boostStrength=0.0,
        columnDimensions=[output_columns],
        inputDimensions=[input_columns * cells_per_column],
        potentialRadius=input_columns * cells_per_column,
        dutyCyclePeriod=1000,
        globalInhibition=True,
        localAreaDensity=0.005,
        minPctOverlapDutyCycle=0.001,
        numActiveColumnsPerInhArea=0,
        potentialPct=0.5,
        spVerbosity=0,
        stimulusThreshold=3,
        synPermConnected=0.5,
        synPermActiveInc=0.1,
        synPermInactiveDec=0.01,
        wrapAround=True
    )

    config = {'tm': config_tm, 'tp': config_tp, 'n_policies': n_policies, 'epochs': epochs, 'threshold': threshold,
              'randomness': randomness, 'n_actions': n_actions, 'action_bucket': action_bucket, 'n_states': n_states,
              'state_bucket': state_bucket, 'input_columns': input_columns, 'cells_per_column': cells_per_column,
              'output_columns': output_columns, 'output_union_sparsity': output_union_sparsity, 'noise_tolerance_apical': noise_tolerance_apical,
              'learning_margin_apical': learning_margin_apical, 'seeds': seeds}
    wandb.init(project='test_cc', entity='hauska', config=config)
    config['log_codes'] = True
    if len(seeds) > 1:
        config['log_train'] = False
    else:
        config['log_train'] = True

    retrieval_accuracy = list()
    for seed in seeds:
        config['seed'] = seed
        config['tm']['seed'] = seed
        config['tp']['seed'] = seed
        retrieval_accuracy.append(run_config(config))
    retrieval_accuracy = np.array(retrieval_accuracy)
    wandb.log({'retrieval_accuracy': retrieval_accuracy.mean(), 'retrieval_accuracy_std': retrieval_accuracy.std()})


if __name__ == '__main__':
    main(seeds=[32443, 43422, 19875, 31341, 94827])
