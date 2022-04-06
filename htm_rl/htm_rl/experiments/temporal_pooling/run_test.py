import sys
from typing import Optional

import numpy as np
from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import ensure_absolute_number, safe_divide
from htm_rl.experiments.temporal_pooling.config_utils import make_logger, compile_config
from htm_rl.experiments.temporal_pooling.data_generation import resolve_data_generator
from htm_rl.experiments.temporal_pooling.metrics import (
    symmetric_error, representations_intersection_1
)
from htm_rl.modules.htm.spatial_pooler import UnionTemporalPooler
from htm_rl.modules.htm.temporal_memory import DelayedFeedbackTM


# noinspection PyAttributeOutsideInit
class ExperimentStats:
    def __init__(self):
        self.tp_current_representation = set()
        # self.tp_prev_policy_union = tp.getUnionSDR().copy()
        # self.tp_prev_union = tp.getUnionSDR().copy()
        ...

    def on_policy_change(self):
        # self.tp_prev_policy_union = self.tp_prev_union.copy()
        # self.tp_prev_union = tp.getUnionSDR().copy()

        self.window_size = 1
        self.window_error = 0
        self.whole_active = None
        self.policy_repeat = 0
        self.intra_policy_step = 0

    def on_policy_repeat(self):
        self.intra_policy_step = 0

        # if self.policy_repeat == 2:
        #     self.whole_active = SDR(tp.getUnionSDR().dense.shape)
        #     self.whole_active.dense = np.zeros(tp.getUnionSDR().dense.shape)

        self.policy_repeat += 1

    def on_step(
            self, temporal_memory, temporal_pooler, logger
    ):
        tm_log = self._get_tm_metrics(temporal_memory)
        tp_log = self._get_tp_metrics(temporal_pooler)
        if logger:
            logger.log(tm_log | tp_log)

        # self.window_error += symmetric_error(tp_output, self.tp_prev_union)
        self.intra_policy_step += 1

    # noinspection PyProtectedMember
    def _get_tp_metrics(self, temporal_pooler) -> dict:
        prev_repr = self.tp_current_representation
        curr_repr = set(temporal_pooler.getUnionSDR().sparse)
        self.tp_current_representation = curr_repr

        sparsity = safe_divide(
            len(curr_repr), temporal_pooler._maxUnionCells
        )
        new_cells_pct = safe_divide(
            len(prev_repr & curr_repr),
            temporal_pooler._maxUnionCells
        )

        # my_log['new_cells_percent'] = 1 - representations_intersection_1(
        #     tp.getUnionSDR().dense, prev_dense
        # )
        # my_log['num_in_prev'] = np.count_nonzero(prev_dense)
        # my_log['num_in_curr'] = np.count_nonzero(tp.getUnionSDR().dense)
        #
        # if whole_active is not None:
        #     whole_active.dense = np.logical_or(whole_active.dense, tp.getUnionSDR().dense)
        #     whole_nonzero = np.count_nonzero(whole_active.dense)
        #     my_log['cells_in_whole'] = np.count_nonzero(tp.getUnionSDR().dense) / whole_nonzero
        #
        # if counter % window_size == window_size - 1:
        #     my_log['difference'] = (window_error / window_size)
        #     window_error = 0
        return {
            'tp/new_cells_pct': new_cells_pct
        }

    def _get_tm_metrics(self, temporal_memory) -> dict:
        active_cells: np.ndarray = temporal_memory.get_active_cells()
        predicted_cells: np.ndarray = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.size, active_cells.size)

        return {
            'tm/recall': recall
        }


class Experiment:
    n_policies: int
    epochs: int
    policy_repeats: int
    steps_per_policy: int

    config: dict
    logger: Optional[Run]

    _tp_active_input: SDR
    _tp_predicted_input: SDR

    def __init__(
            self, config: dict, n_policies: int, epochs: int,
            policy_repeats: int, steps_per_policy: int,
            **kwargs
    ):
        self.config = config
        self.logger = make_logger(config)
        self.n_policies = n_policies
        self.epochs = epochs
        self.policy_repeats = policy_repeats
        self.steps_per_policy = steps_per_policy

        print('==> Init')
        self.data_generator = resolve_data_generator(config)
        self.temporal_memory = resolve_tm(
            self.config,
            action_encoder=self.data_generator.action_encoder,
            state_encoder=self.data_generator.state_encoder
        )
        self.temporal_pooler = resolve_tp(
            self.config,
            temporal_memory=self.temporal_memory
        )
        self.stats = ExperimentStats()

        # pre-allocated SDR
        tp_input_size = self.temporal_pooler.getNumInputs()
        self._tp_active_input = SDR(tp_input_size)
        self._tp_predicted_input = SDR(tp_input_size)

    def run(self):
        print('==> Generate policies')
        policies = self.data_generator.generate_policies(self.n_policies)

        print('==> Run')
        for epoch in range(self.epochs):
            self.train_epoch(policies)
        print('<==')

    def train_epoch(self, policies):
        representations = []

        for policy in policies:
            self.temporal_pooler.reset()
            self.stats.on_policy_change()

            for i in range(self.policy_repeats):
                self.stats.on_policy_repeat()
                self.run_policy(policy, learn=True)

            representations.append(self.temporal_pooler.getUnionSDR())
        return representations

    def run_policy(self, policy, learn=True):
        tm, tp = self.temporal_memory, self.temporal_pooler

        for state, action in policy:
            self.compute_tm_step(
                feedforward_input=action,
                basal_context=state,
                apical_feedback=self.temporal_pooler.getUnionSDR().sparse,
                learn=learn
            )
            self.compute_tp_step(
                active_input=tm.get_active_cells(),
                predicted_input=tm.get_correctly_predicted_cells(),
                learn=learn
            )

            current_union = self.temporal_pooler.getUnionSDR().sparse.copy()

            self.stats.on_step(
                temporal_memory=self.temporal_memory,
                temporal_pooler=self.temporal_pooler,
                logger=self.logger
            )

    def compute_tm_step(
            self, feedforward_input: SparseSdr, basal_context: SparseSdr,
            apical_feedback: SparseSdr, learn: bool
    ):
        tm = self.temporal_memory

        tm.set_active_context_cells(basal_context)
        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(apical_feedback)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(feedforward_input)
        tm.activate_cells(learn)

    def compute_tp_step(self, active_input: SparseSdr, predicted_input: SparseSdr, learn: bool):
        self._tp_active_input.sparse = active_input.copy()
        self._tp_predicted_input.sparse = predicted_input.copy()

        self.temporal_pooler.compute(self._tp_active_input, self._tp_predicted_input, learn)

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


def resolve_tp(config, temporal_memory):
    base_config_tp = config['temporal_pooler']
    input_size = temporal_memory.columns * temporal_memory.cells_per_column

    config_tp = dict(
        inputDimensions=[input_size],
        potentialRadius=input_size,
    )

    config_tp = base_config_tp | config_tp
    tp = UnionTemporalPooler(**config_tp)
    return tp


def resolve_tm(config, action_encoder, state_encoder):
    base_config_tm = config['temporal_memory']

    # apical feedback
    apical_feedback_cells = base_config_tm['feedback_cells']
    apical_active_bits = ensure_absolute_number(
        base_config_tm['sample_size_apical'],
        baseline=apical_feedback_cells
    )
    activation_threshold_apical = ensure_absolute_number(
        base_config_tm['activation_threshold_apical'],
        baseline=apical_active_bits
    )
    learning_threshold_apical = ensure_absolute_number(
        base_config_tm['learning_threshold_apical'],
        baseline=apical_active_bits
    )

    # basal context
    basal_active_bits = state_encoder.n_active_bits

    config_tm = dict(
        columns=action_encoder.output_sdr_size,

        feedback_cells=apical_feedback_cells,
        sample_size_apical=apical_active_bits,
        activation_threshold_apical=activation_threshold_apical,
        learning_threshold_apical=learning_threshold_apical,
        max_synapses_per_segment_apical=apical_active_bits,

        context_cells=state_encoder.output_sdr_size,
        sample_size_basal=basal_active_bits,
        activation_threshold_basal=basal_active_bits,
        learning_threshold_basal=basal_active_bits,
        max_synapses_per_segment_basal=basal_active_bits,
    )

    # it's necessary as we shadow some "relative" values with the "absolute" values
    config_tm = base_config_tm | config_tm
    tm = DelayedFeedbackTM(**config_tm)
    return tm


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
