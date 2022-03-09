import numpy as np
from htm.bindings.sdr import SDR
from htm_rl.modules.htm.temporal_memory import ApicalBasalFeedbackTM
from htm.bindings.algorithms import SpatialPooler
from htm_rl.modules.basal_ganglia import BasalGanglia, DualBasalGanglia
from htm_rl.modules.htm.pattern_memory import SpatialMemory

import os
import pickle

from typing import Union

EPS = 1e-12


class Block:
    """
    Processing unit of Hierarchy.
    :param tm: ApicalBasalFeedbackTM
        Temporal memory
    :param sp: SpatialPoller or None
        Spatial poller
    :param bg: BasalGanglia or None
        Basal ganglia
    """
    tm: ApicalBasalFeedbackTM
    sp: SpatialPooler
    bg: Union[BasalGanglia, DualBasalGanglia]
    sm: SpatialMemory

    def __init__(self,
                 tm: ApicalBasalFeedbackTM,
                 sm: SpatialMemory = None,
                 sp: SpatialPooler = None,
                 bg: Union[BasalGanglia, DualBasalGanglia] = None,
                 id_: int = None,
                 level: int = None,
                 predicted_boost: float = 0.2,
                 feedback_boost_range: list[float, float] = None,
                 gamma: float = 0.9,
                 sm_da: float = 0,
                 sm_dda: float = 0,
                 d_an_th: float = 0,
                 d_cn_th: float = 0,
                 sm_reward_inc: float = 0.9,
                 sm_reward_dec: float = 0.999,
                 min_reward_decay: float = 0.99,
                 max_reward_decay: float = 0.99,
                 sm_max_reward: float = 0.9,
                 sm_min_reward: float = 0.9,
                 modulate_tm_lr: bool = False,
                 sparsity: float = 0,
                 continuous_output: bool = False):
        
        self.tm = tm
        self.sp = sp
        self.bg = bg
        self.sm = sm

        if self.sp is not None:
            self.sp_output = SDR(self.sp.getColumnDimensions())
            self.sp_input = SDR(self.sp.getInputDimensions())
        else:
            self.sp_output = None
            self.sp_input = SDR(self.tm.basal_columns)

        if feedback_boost_range is None:
            self.feedback_boost_range = [0, 1]
        else:
            self.feedback_boost_range = feedback_boost_range

        self.predicted_columns = SDR(self.tm.basal_columns)

        self.basal_columns = tm.basal_columns
        
        self.basal_in = list()
        self.apical_in = list()
        self.feedback_in = list()

        self.basal_out = list()
        self.apical_out = list()
        self.feedback_out = list()

        self.d_an_th = d_an_th
        self.d_cn_th = d_cn_th

        self.anomaly = -1
        self.confidence = 1
        self.anomaly_threshold = 0
        self.confidence_threshold = 0
        self.d_an = 0
        self.d_cn = 0

        self.da = 0
        self.dda = 0
        self.sm_da = sm_da
        self.sm_dda = sm_dda

        self.modulate_tm_lr = modulate_tm_lr
        self.reward_modulation_signal = 1
        self.sm_reward_inc = sm_reward_inc
        self.sm_reward_dec = sm_reward_dec
        self.mean_reward = 0
        self.max_reward = 0
        self.min_reward = 0
        self.max_reward_decay = max_reward_decay
        self.min_reward_decay = min_reward_decay
        self.sm_max_reward = sm_max_reward
        self.sm_min_reward = sm_min_reward

        self.should_return_exec_predictions = False
        self.should_return_apical_predictions = False

        self.id = id_
        self.level = level

        self.feedback_in_pattern = np.empty(0)
        self.apical_in_pattern = np.empty(0)
        self.basal_in_pattern = np.empty(0)
        self.feedback_in_size = 0
        self.apical_in_size = 0
        self.basal_in_size = 0
        self.sparsity = sparsity

        self.reward_ext = 0
        self.reward_int = 0
        self.k = 0
        self.gamma = gamma
        self.made_decision = False
        self.current_option = None
        self.failed_option = None
        self.completed_option = None
        self.predicted_options = None

        self.predicted_boost = predicted_boost
        self.feedback_boost = 0

        self.learn_tm = True
        self.learn_sp = True
        self.learn_sm = True

        self.continuous_output = continuous_output

    def __str__(self):
        return f"Block_{self.id}"

    def compute(self, add_exec=False, learn_exec=False, narrow_prediction=False):
        self.should_return_exec_predictions = False
        self.should_return_apical_predictions = False
        # gather all inputs
        # form basal input sdr(columns)
        if learn_exec:
            if self.learn_tm:
                feedback_active_columns = list()
                shift = 0

                for block in self.feedback_in:
                    feedback_active_columns.append(block.get_output('basal') + shift)
                    shift += block.basal_columns

                if len(feedback_active_columns) > 0:
                    feedback_active_columns = np.concatenate(feedback_active_columns)
                else:
                    feedback_active_columns = np.empty(0)

                self.tm.set_active_feedback_columns(feedback_active_columns)
                self.tm.activate_exec_dendrites()
                self.tm.learn_exec_feedback_segments()

                self.feedback_in_pattern = feedback_active_columns
                self.apical_in_pattern = np.empty(0)
                self.basal_in_pattern = np.empty(0)

                self.tm.inactivate_exec_dendrites()

            self.d_an = 0
            self.d_cn = 0
        elif add_exec:
            self.should_return_exec_predictions = True
            feedback_active_columns = list()
            shift = 0
            total_value = 0

            for block in self.feedback_in:
                pattern, value = block.get_output('feedback', return_value=True)
                feedback_active_columns.append(pattern + shift)
                shift += block.basal_columns
                if value is not None:
                    total_value += value

            if len(self.feedback_in) > 1:
                total_value /= len(self.feedback_in)

            if len(feedback_active_columns) > 0:
                feedback_active_columns = np.concatenate(feedback_active_columns)
            else:
                feedback_active_columns = np.empty(0)

            self.tm.set_active_feedback_columns(feedback_active_columns)
            self.tm.activate_exec_dendrites()

            self.feedback_in_pattern = feedback_active_columns

            self.d_an = 0
            self.d_cn = 0

            # Evaluate feedback boost
            self.feedback_boost = self.feedback_boost_range[0] + total_value * (self.feedback_boost_range[1] - self.feedback_boost_range[0])
        elif narrow_prediction:
            # Narrow prediction by a feedback
            # Form feedback input sdr(columns)
            feedback_active_columns = list()
            shift = 0

            for block in self.feedback_in:
                feedback_active_columns.append(block.get_output('basal') + shift)
                shift += block.basal_columns

            if len(feedback_active_columns) > 0:
                feedback_active_columns = np.concatenate(feedback_active_columns)
            else:
                feedback_active_columns = np.empty(0)

            # TM
            self.tm.set_active_feedback_columns(feedback_active_columns)
            self.tm.activate_inhib_dendrites()
            self.tm.predict_cells()

            self.confidence = self.tm.confidence[-1]
            self.confidence_threshold = self.tm.confidence_threshold

            self.d_cn = (self.confidence - self.confidence_threshold) / (self.confidence_threshold + EPS)

            self.feedback_in_pattern = feedback_active_columns
        else:
            basal_active_columns = list()
            shift = 0

            for block in self.basal_in:
                basal_active_columns.append(block.get_output('basal') + shift)
                shift += block.basal_columns

            if len(basal_active_columns) > 0:
                basal_active_columns = np.concatenate(basal_active_columns)
            else:
                basal_active_columns = np.empty(0)

            # Form apical input sdr(cells)
            apical_active_cells = list()
            apical_winner_cells = list()
            shift = 0

            for block in self.apical_in:
                active, winner = block.get_output('apical')
                apical_active_cells.append(active + shift)
                apical_winner_cells.append(winner + shift)
                shift += block.tm.basal_total_cells

            if len(apical_active_cells) > 0:
                apical_active_cells = np.concatenate(apical_active_cells)
            else:
                apical_active_cells = np.empty(0)

            if len(apical_winner_cells) > 0:
                apical_winner_cells = np.concatenate(apical_winner_cells)
            else:
                apical_winner_cells = np.empty(0)

            # Form feedback input sdr(columns)
            feedback_active_columns = list()
            shift = 0

            for block in self.feedback_in:
                feedback_active_columns.append(block.get_output('basal') + shift)
                shift += block.basal_columns

            if len(feedback_active_columns) > 0:
                feedback_active_columns = np.concatenate(feedback_active_columns)
            else:
                feedback_active_columns = np.empty(0)

            # SP
            self.sp_input.sparse = basal_active_columns
            if self.sp is not None:
                self.sp.compute(self.sp_input, self.learn_sp, self.sp_output)
                basal_active_columns = self.sp_output.sparse
            # Refresh patterns
                if (self.sm is not None) and self.learn_sm:
                    self.sm.add(self.sp_output.dense.copy())
            else:
                if (self.sm is not None) and self.learn_sm:
                    self.sm.add(self.sp_input.dense.copy())

            # Reinforce
            if (self.bg is not None) and (self.k != 0):
                self.bg.update_response(basal_active_columns)
                self.bg.force_dopamine(self.reward_ext, k=self.k, reward_int=self.reward_int)

                self.update_reward_modulation_signal(self.reward_ext)

                self.reward_ext = 0
                self.reward_int = 0
                self.k = 0

                prev_da = self.da
                self.da = self.da * self.sm_da + np.power(self.bg.td_error, 2).flatten().sum() * (1 - self.sm_da)
                self.dda = self.dda * self.sm_dda + (self.da - prev_da) * (1 - self.sm_dda)

            # Forgetting
            if (self.sm is not None) and self.learn_sm:
                self.sm.forget()

            # Modulation
            if self.modulate_tm_lr:
                self.tm.set_learning_rate(self.reward_modulation_signal)
            # TM
            self.tm.set_active_columns(basal_active_columns)
            self.tm.activate_cells(self.learn_tm)

            self.anomaly = self.tm.anomaly[-1]
            self.anomaly_threshold = self.tm.anomaly_threshold

            self.d_an = (self.anomaly - self.anomaly_threshold)/(self.anomaly_threshold + EPS)

            self.tm.set_active_apical_cells(apical_active_cells)
            self.tm.set_winner_apical_cells(apical_winner_cells)
            self.tm.set_active_feedback_columns(feedback_active_columns)

            self.tm.activate_basal_dendrites()
            self.tm.activate_apical_dendrites()
            self.tm.activate_inhib_dendrites()

            self.tm.predict_cells()

            self.confidence = self.tm.confidence[-1]
            self.confidence_threshold = self.tm.confidence_threshold

            self.d_cn = (self.confidence - self.confidence_threshold)/(self.confidence_threshold + EPS)

            self.feedback_in_pattern = feedback_active_columns
            self.apical_in_pattern = apical_active_cells
            self.basal_in_pattern = basal_active_columns

    def get_output(self, mode, return_value=False):
        """
        Get block output.
        :param mode: str: type of output, modes: {'basal', 'apical', 'feedback'}
        :return: depends on mode
        """
        if mode == 'basal':
            # active columns without filtration
            return self.tm.get_active_columns()
        elif mode == 'apical':
            # apical active cells and winners
            return self.tm.get_active_cells(), self.tm.get_winner_cells()
        elif mode == 'feedback':
            # basal predicted columns with filtration
            predicted_columns = self.tm.get_predicted_columns(add_exec=self.should_return_exec_predictions,
                                                              add_apical=self.should_return_apical_predictions)
            self.predicted_columns.sparse = predicted_columns
            # filter columns by Basal Ganglia conditioned on apical input
            if (self.bg is not None) and (self.sm is not None):
                # form apical input
                apical_active_columns = list()
                shift = 0

                for block in self.apical_in:
                    columns = block.get_output('basal')
                    apical_active_columns.append(columns + shift)
                    shift += block.basal_columns

                if len(apical_active_columns) > 0:
                    apical_active_columns = np.concatenate(apical_active_columns)
                else:
                    apical_active_columns = np.empty(0)

                condition = SDR(shift)
                condition.sparse = apical_active_columns

                # detect options among predictions
                predicted_options, indices = self.sm.get_options(self.predicted_columns.dense, return_indices=True)
                # all options
                options = self.sm.get_sparse_patterns()

                if len(options) > 0 or self.continuous_output:
                    boost_predicted_options = np.zeros(len(self.sm))
                    if len(indices) > 0:
                        # boost predicted options
                        boost_predicted_options[indices] += self.predicted_boost
                        # feedback boost
                        boost_predicted_options[indices] += self.feedback_boost

                    option_index, option, option_values = self.bg.compute(condition.sparse, options, responses_boost=boost_predicted_options)
                    self.bg.update_stimulus(condition.sparse)

                    norm_option_values = option_values - option_values.min()
                    norm_option_values /= (norm_option_values.max() + EPS)

                    self.made_decision = True
                    self.failed_option = None
                    self.completed_option = None
                    if len(self.sm.unique_id) > 0:
                        self.current_option = self.sm.unique_id[option_index]
                        self.predicted_options = self.sm.unique_id[indices]
                    else:
                        self.current_option = None
                        self.predicted_options = np.empty(0)

                    # jumped off a high level option
                    if not np.isin(option_index, indices):
                        self.feedback_boost = 0
                        for block in self.feedback_in:
                            block.finish_current_option('failed')

                    if return_value:
                        return option, norm_option_values[option_index]
                    else:
                        return option
                else:
                    if return_value:
                        return np.empty(0), None
                    else:
                        return np.empty(0)
            else:
                if return_value:
                    return predicted_columns, None
                else:
                    return predicted_columns
        else:
            raise ValueError(f'There is no such mode {mode}!')

    def get_in_sizes(self):
        self.feedback_in_size = sum([block.basal_columns for block in self.feedback_in])
        self.apical_in_size = sum([block.tm.basal_total_cells for block in self.apical_in])
        self.basal_in_size = sum([block.basal_columns for block in self.basal_in])
        return self.feedback_in_size, self.apical_in_size, self.basal_in_size

    def add_reward(self, reward_ext: float, reward_int: float = 0):
        if self.bg is not None:
            self.reward_ext += (self.gamma ** self.k) * reward_ext
            self.reward_int += (self.gamma ** self.k) * reward_int
            self.k += 1

    def finish_current_option(self, flag):
        if flag == 'failed':
            self.failed_option = self.current_option
            self.completed_option = None
        elif flag == 'completed':
            self.completed_option = self.current_option
            self.failed_option = None
        else:
            raise ValueError

        self.made_decision = False
        self.current_option = None

    def reset(self):
        self.tm.reset()

        if self.bg is not None:
            self.bg.reset()

        self.reward_ext = 0
        self.reward_int = 0
        self.k = 0
        self.made_decision = False
        self.current_option = None
        self.failed_option = None
        self.completed_option = None

        self.should_return_exec_predictions = False
        self.should_return_apical_predictions = False

        self.feedback_in_pattern = np.empty(0)
        self.apical_in_pattern = np.empty(0)
        self.basal_in_pattern = np.empty(0)

    def freeze(self):
        self.learn_sp = False
        self.learn_tm = False
        self.learn_sm = False

    def unfreeze(self):
        self.learn_sp = True
        self.learn_tm = True
        self.learn_sm = True

    def update_reward_modulation_signal(self, reward):
        if reward > self.mean_reward:
            self.mean_reward = self.mean_reward * self.sm_reward_inc + reward * (1 - self.sm_reward_inc)
        else:
            self.mean_reward = self.mean_reward * self.sm_reward_dec + reward * (1 - self.sm_reward_dec)

        if self.mean_reward > self.max_reward:
            self.max_reward = self.max_reward * self.sm_max_reward + self.mean_reward * (1 - self.sm_max_reward)
        else:
            self.max_reward *= self.max_reward_decay

        if self.mean_reward < self.min_reward:
            self.min_reward = self.min_reward * self.sm_min_reward + self.mean_reward * (1 - self.sm_min_reward)
        else:
            self.min_reward *= self.min_reward_decay

        if abs(self.max_reward) < EPS:
            self.reward_modulation_signal = 0
        else:
            self.reward_modulation_signal = np.clip((self.mean_reward - self.min_reward) / self.max_reward, 0.0,
                                                    1.0)


class InputBlock:
    """
    Dummy block for input patterns
    :param columns: int
    Number of input columns
    """
    def __init__(self, columns, id_=None, level=None, sparsity=None):
        self.pattern = np.empty(0)

        self.basal_columns = columns
        self.sparsity = sparsity
        self.basal_in = list()
        self.apical_in = list()
        self.feedback_in = list()

        self.basal_out = list()
        self.apical_out = list()
        self.feedback_out = list()

        self.id = id_
        self.level = level

        self.learn_sp = False
        self.learn_tm = False
        self.bg = None
        self.tm = None
        self.sp = None
        self.anomaly = 0
        self.anomaly_threshold = 0
        self.confidence = 0
        self.confidence_threshold = 0
        self.d_an = 0
        self.d_cn = 0
        self.d_an_th = 0
        self.d_cn_th = 0
        self.freq = 0
        self.reward_modulation_signal = 1

    def __str__(self):
        return f"InputBlock_{self.id}"

    def reinforce(self, *args, **kwargs):
        pass

    def get_output(self, *args, **kwargs):
        return self.pattern

    def set_pattern(self, pattern):
        self.pattern = pattern

    def get_in_sizes(self):
        return 0, 0, 0

    def add_reward(self, reward_ext, reward_int=0):
        pass

    def reset(self):
        self.pattern = np.empty(0)

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class Hierarchy:
    """
    Compute hierarchy of blocks
    :param blocks: list(Block)
    :param input_blocks: list of indices of blocks
    :param output_block: index of output block
    :param visual_block: index of visual block
    :param block_connections: list
    List of dicts. Every dict corresponds to block and describes its connections.
    block id corresponds to dict position in block_connections
    Example:
    [{'apical_in': [2, 3],
      'basal_in': [4],
      'feedback_in: []',
      'apical_out': [2, 3],
      'basal_out': [1],
      'feedback_out: []'},
      {'apical_in': [0, 3],
      'basal_in': [5],
      'feedback_in: []',
      ...},
      ...
    ]
    """
    output_block: Block

    def __init__(self, blocks: list, input_blocks: list, output_block: int, visual_block: int, block_connections: list, logs_dir=None):
        self.queue = list()
        self.blocks = blocks
        self.input_blocks = [blocks[i] for i in input_blocks]
        self.output_block = blocks[output_block]
        self.visual_block = blocks[visual_block]
        self.block_connections = block_connections
        self.block_sizes = list()
        self.block_levels = list()

        # wire blocks together
        for block_id, connections, block in zip(range(len(blocks)), block_connections, blocks):
            block.apical_in = [blocks[i] for i in connections['apical_in']]
            block.basal_in = [blocks[i] for i in connections['basal_in']]
            block.feedback_in = [blocks[i] for i in connections['feedback_in']]

            block.apical_out = [blocks[i] for i in connections['apical_out']]
            block.basal_out = [blocks[i] for i in connections['basal_out']]
            block.feedback_out = [blocks[i] for i in connections['feedback_out']]

            block.id = block_id
            feedback_in_size, apical_in_size, basal_in_size = block.get_in_sizes()
            basal_out_size = block.basal_columns
            self.block_sizes.append({'feedback_in': feedback_in_size,
                                     'apical_in': apical_in_size,
                                     'basal_in': basal_in_size,
                                     'basal_out': basal_out_size})
            self.block_levels.append(block.level)

        # logging
        self.logs_dir = logs_dir
        self.logs = {'tasks': list(),
                     'patterns': {'feedback_in': list(),
                                  'basal_out': list()},
                     'anomaly': list(),
                     'confidence': list()}

        if logs_dir is not None:
            with open(os.path.join(logs_dir, 'info.pkl'), 'wb') as file:
                pickle.dump({'connections': block_connections,
                             'input_blocks': [block.id for block in input_blocks],
                             'block_sizes': self.block_sizes,
                             'block_levels': self.block_levels},
                            file)

    def compute(self):
        block, kwargs = self.queue.pop()

        if kwargs is None:
            block.compute()
        else:
            block.compute(**kwargs)

        if (block.d_an > block.d_an_th) and (block.d_cn >= - block.d_cn_th):
            tasks = zip(block.basal_out, [None]*len(block.basal_out))
            self.queue.append((block, {'narrow_prediction': True}))
            self.queue.append((block, {'learn_exec': True}))
            self.queue.extend(tasks)
        elif block.d_cn < -block.d_cn_th:
            self.queue.append((block, {'add_exec': True}))
            # end of an option
            if block.anomaly <= block.anomaly_threshold:
                for block in block.feedback_in:
                    block.finish_current_option('completed')

        # logging
        if self.logs_dir is not None:
            self.logs['tasks'].append((block.id, kwargs))
            self.logs['patterns']['basal_out'].append(block.get_output('basal'))
            self.logs['patterns']['feedback_in'].append(block.feedback_in_pattern)
            self.logs['anomaly'].append((block.anomaly, block.anomaly_threshold))
            self.logs['confidence'].append((block.confidence, block.confidence_threshold))

        if len(self.queue) > 0:
            self.compute()

    def set_input(self, patterns):
        """
        Set patterns to input blocks
        :param patterns: list of lists of active bits or None
        None if there is no pattern for specific input
        For example: [None, [1, 2, 3], [4, 5, 6]]
        So there is no pattern for the first input block
        :return:
        """
        for pattern, block in zip(patterns, self.input_blocks):
            # logging
            if self.logs_dir is not None:
                self.logs['tasks'].append((block.id, {'input_block': True}))
                self.logs['patterns']['basal_out'].append(pattern)
                self.logs['patterns']['feedback_in'].append(np.empty(0))
                self.logs['anomaly'].append(None)
                self.logs['confidence'].append(None)

            if pattern is not None:
                block.set_pattern(pattern)
                tasks = list(zip(block.basal_out, [None]*len(block.basal_out)))
                self.queue.extend(tasks)
        self.queue = self.queue[::-1]
        # start processing input
        self.compute()

    def add_rewards(self, rewards_ext, rewards_int):
        for reward_ext, reward_int, block in zip(rewards_ext, rewards_int, self.blocks):
            block.add_reward(reward_ext, reward_int)

    def save_logs(self):
        if self.logs_dir is not None:
            with open(os.path.join(self.logs_dir, 'logs.pkl'), 'wb') as file:
                pickle.dump(self.logs,
                            file)
        else:
            raise ValueError('Log dir is not defined!')

    def reset(self):
        for block in self.blocks:
            block.reset()

    def freeze(self):
        for block in self.blocks:
            block.freeze()

    def unfreeze(self):
        for block in self.blocks:
            block.unfreeze()
