import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from htm.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

from htm_rl.experiments.temporal_pooling.config import *
from htm_rl.experiments.temporal_pooling.custom_utp import CustomUtp
from htm_rl.experiments.temporal_pooling.data_generation import generate_data
from htm_rl.experiments.temporal_pooling.metrics import (
    symmetric_error, representations_intersection_1, row_similarity,
    representation_similarity
)
from htm_rl.experiments.temporal_pooling.sandwich_tp import SandwichTp
from htm_rl.experiments.temporal_pooling.utils import StupidEncoder
from htm_rl.modules.htm.spatial_pooler import SpatialPooler
from htm_rl.modules.htm.spatial_pooler import UnionTemporalPooler
from htm_rl.modules.htm.temporal_memory import DelayedFeedbackTM


def learn_model(tm: TemporalMemory, sdrs: np.ndarray, num_epochs=10) -> list:
    errors = []
    for epoch in range(num_epochs):
        for sdr in sdrs:
            tm.compute(sdr, learn=True)
            tm.activateDendrites(True)
            errors.append(tm.anomaly)
        tm.compute(SDR(sdrs[0].dense.shape), learn=False)
    return errors


def run(tm, tp, policy, state_encoder, action_encoder, learn=True, prev_dense=None, whole_active: SDR = None):
    tp_prev_union = tp.getUnionSDR().sparse.copy()
    tp_input = SDR(tp.getNumInputs())
    tp_predictive = SDR(tp.getNumInputs())
    window_size = 1
    window_error = 0
    counter = 0
    for state, action in policy:
        context = state_encoder.encode(state)
        active_input = action_encoder.encode(action)

        tm.set_active_context_cells(context)

        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(tp.getUnionSDR().sparse)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(active_input)
        tm.activate_cells(learn)

        tp_input.sparse = tm.get_active_cells()
        tp_predictive.sparse = tm.get_correctly_predicted_cells()
        tp.compute(tp_input, tp_predictive, learn)

        current_union = tp.getUnionSDR().sparse.copy()

        window_error += symmetric_error(current_union, tp_prev_union)

        my_log = {}
        if prev_dense is not None:
            my_log['new_cells_percent'] = 1 - representations_intersection_1(tp.getUnionSDR().dense, prev_dense)
            my_log['num_in_prev'] = np.count_nonzero(prev_dense)
            my_log['num_in_curr'] = np.count_nonzero(tp.getUnionSDR().dense)

        if whole_active is not None:
            whole_active.dense = np.logical_or(whole_active.dense, tp.getUnionSDR().dense)
            whole_nonzero = np.count_nonzero(whole_active.dense)
            my_log['cells_in_whole'] = np.count_nonzero(tp.getUnionSDR().dense) / whole_nonzero

        if counter % window_size == window_size - 1:
            my_log['difference'] = (window_error / window_size)
            window_error = 0
        wandb.log(my_log)
        tp_prev_union = current_union.copy()

        counter += 1


def run_only_tp(_tp, sequence, _encoder, learn=True):
    prev_union = _tp.getUnionSDR().sparse.copy()

    for element in sequence:
        _input = _encoder.encode(element)
        sdr = SDR(_tp.getNumInputs())
        sdr.dense = _input
        _tp.compute(sdr, sdr, True)
        diff = symmetric_error(prev_union, _tp.getUnionSDR().sparse)
        wandb.log({'difference': diff})


def train_all_seq(tm, tp, data, state_encoder, action_encoder, iters_per_seq):
    representations = []
    prev = tp.getUnionSDR().dense.copy()
    for policy in data:
        tp.reset()
        whole_active = np.zeros(tp.getUnionSDR().dense.shape)
        for i in range(iters_per_seq):
            if i < 2:
                whole_active = None
            elif i == 2:
                whole_active = SDR(tp.getUnionSDR().dense.shape)
                whole_active.dense = np.zeros(tp.getUnionSDR().dense.shape)
            run(tm, tp, policy, state_encoder, action_encoder, learn=True, prev_dense=prev, whole_active=whole_active)
        representations.append(tp.getUnionSDR())
        prev = tp.getUnionSDR().dense.copy()
    return representations


def _run_tests():
    wandb.login()
    row_data, data = generate_data(5, n_actions, n_states, randomness=0.5)
    np.random.shuffle(data)
    print(data)

    # -----------------------------
    errors = []
    wandb.init(project='my_utp', entity='irodkin', reinit=True)
    tm = DelayedFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)

    for i in range(100):
        run(tm, tp, data[0], state_encoder, action_encoder, True)
    wandb.finish()

    # -----------------------------
    tm = DelayedFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)

    # -----------------------------

    utp_conf = {
        'inputDimensions': [input_columns * cells_per_column],
        'columnDimensions': [output_columns],
        'initial_pooling': 1,
        'pooling_decay': 0.1,
        'permanence_inc': 0.005,
        'permanence_dec': 0.003,
        'sparsity': 0.04,
        'active_weight': 0.5,
        'predicted_weight': 2.0,
        'receptive_field_sparsity': 0.5,
        'activation_threshold': 0.6,
    }
    np.random.seed(42)
    tp_shape = utp_conf['columnDimensions']
    in_shape = utp_conf['inputDimensions']
    prev = np.zeros(utp_conf['columnDimensions'])
    classes_num = utp_conf['inputDimensions']

    my_utp = CustomUtp(**utp_conf)

    wandb.init(project='my_utp', entity='irodkin', reinit=True, config=utp_conf)

    # -----------------------------
    my_utp = CustomUtp(**utp_conf)
    tm = DelayedFeedbackTM(**config_tm)

    for i in range(100):
        run(tm, my_utp, data[0], state_encoder, action_encoder, True)

    wandb.finish(quiet=True)


    # -----------------------------
    plt.title('pooling activations')
    sns.heatmap(my_utp._pooling_activations.reshape(-1, 80), vmin=0, vmax=1, cmap='plasma')

    # -----------------------------
    print(my_utp.getUnionSDR().dense.nonzero()[0].size / output_columns)

    # -----------------------------
    wandb.init(project='my_utp', entity='irodkin', reinit=True, config=utp_conf)

    my_utp = CustomUtp(**utp_conf)

    for i in range(100):
        run_only_tp(my_utp, row_data[0], StupidEncoder(n_actions, utp_conf['inputDimensions'][0]), True)

    wandb.finish(quiet=True)

    # -------------

    # ------------------
    utp_conf = {
        'inputDimensions': [input_columns * cells_per_column],
        'columnDimensions': [output_columns],
        'initial_pooling': 1,
        'pooling_decay': 0.2,
        'permanence_inc': 0.005,
        'permanence_dec': 0.003,
        'sparsity': 0.004,
        'active_weight': 0.5,
        'predicted_weight': 1.0,
        'receptive_field_sparsity': 0.5,
        'activation_threshold': 0.6,
        **config_sp_lower
    }

    wandb.init(project='my_utp', entity='irodkin', reinit=True, config=utp_conf)

    my_utp = CustomUtp(**utp_conf)
    tm = DelayedFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)
    for epoch in range(5):
        representations = train_all_seq(tm, my_utp, data, state_encoder, action_encoder, 20)

    wandb.finish(quiet=True)

    # -----------------------


    similarity_matrix = np.zeros((len(representations), len(representations)))
    pure_similarity = np.zeros(similarity_matrix.shape)
    for i, policy1 in enumerate(data):
        for j, policy2 in enumerate(data):
            pure_similarity[i][j] = row_similarity(policy1, policy2)
            similarity_matrix[i][j] = abs(
                representation_similarity(representations[i].dense, representations[j].dense)
            )

    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    sns.heatmap(similarity_matrix, vmin=0, vmax=1, cmap='plasma', ax=ax1)
    sns.heatmap(pure_similarity, vmin=0, vmax=1, cmap='plasma', ax=ax2)

    sns.heatmap(abs(pure_similarity - similarity_matrix), vmin=0, vmax=1, cmap='plasma', ax=ax3)
    plt.show()
    # print(representations[0].dense.nonzero()[0])
    # print(representations[1].dense.nonzero()[0])

    # ----------------------------

    # ------------------
    stp_config = dict(
        initial_pooling=1,
        pooling_decay=0.05,
        lower_sp_conf=config_sp_lower,
        upper_sp_conf=config_sp_upper
    )

    tm = DelayedFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)
    stp = SandwichTp(**stp_config)

    wandb.init(project='my_utp', entity='irodkin', reinit=True, config=stp_config)
    for epoch in range(50):
        representations = train_all_seq(tm, stp, data, state_encoder, action_encoder, 20)
    wandb.finish(quiet=True)

    # -----------------
    sp1 = SpatialPooler(**config_sp_lower)
    sp2 = SpatialPooler(**config_sp_upper)

    # -----------------
    input = SDR(sp1.getNumInputs())
    input.dense = np.ones(sp1.getNumInputs())
    output = SDR(sp1.getColumnDimensions())
    output.dense = np.ones(sp1.getColumnDimensions())

    sp1.compute(input, learn=True, output=output)
    sp2.compute(output, learn=True, output=output)
    print(output.sparse)


if __name__ == '__main__':
    _run_tests()
