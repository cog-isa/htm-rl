import sys
from typing import Optional

import numpy as np
import wandb
import yaml
from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from htm_rl.experiments.temporal_pooling.data_generation import resolve_data_generator
from htm_rl.experiments.temporal_pooling.metrics import symmetric_error, representations_intersection_1
from htm_rl.modules.htm.spatial_pooler import UnionTemporalPooler
from htm_rl.modules.htm.temporal_memory import DelayedFeedbackTM
from htm_rl.scenarios.utils import parse_str


class Experiment:
    n_policies: int
    epochs: int
    steps_per_policy: int

    config: dict
    logger: Optional[Run]

    def __init__(self, config: dict, n_policies: int, epochs: int, steps_per_policy: int, **kwargs):
        self.config = config
        self.logger = make_logger(config)
        self.n_policies = n_policies
        self.epochs = epochs
        self.steps_per_policy = steps_per_policy

    def run(self):
        config = self.config
        print(config)
        data_generator = resolve_data_generator(config)

        policies = data_generator.generate_policies(self.n_policies)
        print(policies)

        tm = resolve_tm(self.config, data_generator.action_encoder, data_generator.state_encoder)
        tp = resolve_tp(self.config, data_generator.action_encoder)

        representations = []
        for epoch in range(self.epochs):
            representations = self.train_epoch(tm, tp, policies)

    def train_epoch(self, tm, tp, policies):
        representations = []
        prev = tp.getUnionSDR().dense.copy()

        for policy in policies:
            tp.reset()
            whole_active = None
            for i in range(self.steps_per_policy):
                if i == 2:
                    whole_active = SDR(tp.getUnionSDR().dense.shape)
                    whole_active.dense = np.zeros(tp.getUnionSDR().dense.shape)

                self.run_one_policy(
                    tm, tp, policy, learn=True, prev_dense=prev, whole_active=whole_active
                )

            representations.append(tp.getUnionSDR())
            prev = tp.getUnionSDR().dense.copy()
        return representations

    def run_one_policy(
            self, tm, tp, policy,
            learn=True, prev_dense=None, whole_active: SDR = None
    ):
        tp_prev_union = tp.getUnionSDR().sparse.copy()
        tp_input = SDR(tp.getNumInputs())
        tp_predictive = SDR(tp.getNumInputs())
        window_size = 1
        window_error = 0
        counter = 0
        for context, active_input in policy:
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

            if self.logger:
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
                self.logger.log(my_log)
            tp_prev_union = current_union.copy()

            counter += 1

    # def vis_what(self, data, representations):
    #     similarity_matrix = np.zeros((len(representations), len(representations)))
    #     pure_similarity = np.zeros(similarity_matrix.shape)
    #     for i, policy1 in enumerate(data):
    #         for j, policy2 in enumerate(data):
    #             pure_similarity[i][j] = row_similarity(policy1, policy2)
    #             similarity_matrix[i][j] = abs(
    #                 representation_similarity(representations[i].dense, representations[j].dense)
    #             )
    #
    #     fig = plt.figure(figsize=(40, 10))
    #     ax1 = fig.add_subplot(131)
    #     ax1.set_title('representational', size=40)
    #     ax2 = fig.add_subplot(132)
    #     ax2.set_title('pure', size=40)
    #     ax3 = fig.add_subplot(133)
    #     ax3.set_title('difference', size=40)
    #
    #     sns.heatmap(similarity_matrix, vmin=0, vmax=1, cmap='plasma', ax=ax1)
    #     sns.heatmap(pure_similarity, vmin=0, vmax=1, cmap='plasma', ax=ax2)
    #
    #     sns.heatmap(abs(pure_similarity - similarity_matrix), vmin=0, vmax=1, cmap='plasma', ax=ax3, annot=True)
    #     wandb.log({'representations similarity': wandb.Image(ax1)})
    #     plt.show()


def resolve_tp(config, action_encoder):
    seed = config['seed']
    input_columns = action_encoder.output_sdr_size
    cells_per_column = 16
    output_columns = 4000
    config_sp_lower = dict(
        boostStrength=0.0,
        columnDimensions=[output_columns],
        inputDimensions=[input_columns * cells_per_column],
        potentialRadius=input_columns * cells_per_column,
        dutyCyclePeriod=1000,
        globalInhibition=True,
        localAreaDensity=0.01,
        minPctOverlapDutyCycle=0.001,
        numActiveColumnsPerInhArea=0,
        potentialPct=0.5,
        spVerbosity=0,
        stimulusThreshold=3,
        synPermConnected=0.5,
        synPermActiveInc=0.1,
        synPermInactiveDec=0.01,
        wrapAround=True,
        seed=seed
    )
    output_union_sparsity = 0.01
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
        **config_sp_lower
    )
    tp = UnionTemporalPooler(**config_tp)
    return tp


def resolve_tm(config, action_encoder, state_encoder):
    input_columns = action_encoder.output_sdr_size
    cells_per_column = 16
    output_columns = 4000
    output_union_sparsity = 0.01
    noise_tolerance_apical = 0.1
    learning_margin_apical = 0.2
    seed = 42
    config_tm = dict(
        columns=input_columns,
        cells_per_column=cells_per_column,
        context_cells=state_encoder.output_sdr_size,
        feedback_cells=output_columns,
        activation_threshold_basal=state_encoder.n_active_bits,
        learning_threshold_basal=state_encoder.n_active_bits,
        activation_threshold_apical=int(
            output_union_sparsity * output_columns * (1 - noise_tolerance_apical)
        ),
        learning_threshold_apical=int(
            output_union_sparsity * output_columns * (1 - learning_margin_apical)
        ),
        connected_threshold_basal=0.5,
        permanence_increment_basal=0.1,
        permanence_decrement_basal=0.01,
        initial_permanence_basal=0.4,
        predicted_segment_decrement_basal=0.001,
        sample_size_basal=state_encoder.n_active_bits,
        max_synapses_per_segment_basal=state_encoder.n_active_bits,
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
        sm_ac=0.99,
        seed=42
    )
    tm = DelayedFeedbackTM(**config_tm)
    return tm


def overwrite_config(config: dict, key_path: str, value: str):
    # accepts everything non-parseable as is, i.e as a str
    value = parse_str(value)
    key_path = key_path.lstrip('-')

    # NOTE: to distinguish sweep params from the config params in wandb
    # interface, we introduced a trick - it's allowed to specify sweep param
    # with insignificant additional dots (e.g. `.path..to...key.`)
    # We ignore them here while parsing the hierarchical path stored in the key.

    # ending dots are removed first to guarantee that after split by dots
    # the last item is the actual correct name stored in the config dict slice
    while key_path.endswith('.'):
        key_path = key_path[:-1]

    # sequentially unfold config dict hierarchy (with the current
    # dict root represented by `c`) following the path stored in the key
    tokens = key_path.split('.')
    c = config
    for key in tokens[:-1]:
        if not key:
            # ignore empty items introduced with additional dots
            continue

        # the sub-key can be integer - an index in an array
        key = parse_str(key)
        # unfold the next level of the hierarchy
        c = c[key]

    # finally, overwrite the value of the last key in the path
    key = parse_str(tokens[-1])
    c[key] = value


def make_logger(config: dict):
    if not config.get('log', None):
        # not specified or empty
        return None

    # TODO: aggregate all wandb-related args into logger['log']
    logger = wandb.init(project=config['project'], entity=config['entity'], config=config)

    return logger


def compile_config(run_args, config_path_prefix: str = '../configs/', config_extension: str = 'yaml'):
    config_name = run_args[0]
    with open(f'{config_path_prefix}{config_name}.{config_extension}', 'r') as config_io:
        config = yaml.load(config_io, Loader=yaml.Loader)

    for arg in run_args[1:]:
        key_path, value = arg.split('.')
        overwrite_config(config, key_path, value)

    return config


def run_test():
    if len(sys.argv) > 1:
        run_args = sys.argv[1:]
    else:
        default_config_name = 'lol'
        run_args = [default_config_name]

    config = compile_config(run_args, config_path_prefix='./configs/')
    Experiment(config, **config['experiment']).run()


if __name__ == '__main__':
    run_test()
