import numpy as np
from htm.bindings.sdr import SDR
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.htm.basal_ganglia import BasalGanglia2
import os
import pickle


class SpatialMemory:
    def __init__(self,
                 overlap_threshold: float,
                 initial_permanence: float = 0.55,
                 permanence_increment: float = 0.1,
                 permanence_decrement: float = 0.05,
                 permanence_threshold: float = 0.5,
                 permanence_forgetting_decrement: float = 0.01,
                 activation_threshold: int = 0):
        self.permanence_forgetting_decrement = permanence_forgetting_decrement
        self.permanence_threshold = permanence_threshold
        self.permanence_decrement = permanence_decrement
        self.permanence_increment = permanence_increment
        self.initial_permanence = initial_permanence
        self.patterns = np.empty(0)
        self.permanence = np.empty(0)
        self.overlap_threshold = overlap_threshold
        self.activation_threshold = activation_threshold

    def __len__(self):
        return self.patterns.shape[0]

    def add(self, dense_pattern: np.array):
        if dense_pattern.sum() >= self.activation_threshold:
            if self.patterns.size == 0:
                self.patterns = dense_pattern.reshape((1, -1))
                self.permanence = np.array([self.initial_permanence])
            else:
                pattern_sizes = self.patterns.sum(axis=1)
                overlaps = 1 - np.sum(np.abs(self.patterns - dense_pattern), axis=1) / (pattern_sizes + 1e-15)

                if np.any(overlaps >= self.overlap_threshold):
                    best_index = np.argmax(overlaps)
                    self.patterns[best_index] = dense_pattern
                    self.permanence[best_index] += self.permanence_increment
                else:
                    self.patterns = np.vstack([self.patterns, dense_pattern.reshape(1, -1)])
                    self.permanence = np.append(self.permanence, self.initial_permanence)

    def reinforce(self, values: np.array):
        """
        Reinforce plausible patterns, forget patterns with permanence under threshold
        :param values: array of values from BG
        :return:
        """
        values -= values.mean()
        values /= (values.std() + 1e-12)
        positive = values > 0
        values[positive] *= self.permanence_increment
        values[~positive] *= self.permanence_decrement

        self.permanence += values
        self.patterns = self.patterns[self.permanence > self.permanence_threshold]
        self.permanence = self.permanence[self.permanence > self.permanence_threshold]

    def forget(self):
        self.permanence -= self.permanence_forgetting_decrement
        self.patterns = self.patterns[self.permanence > self.permanence_threshold]
        self.permanence = self.permanence[self.permanence > self.permanence_threshold]

    def get_options(self, dense_pattern, return_indices=False):
        if self.patterns.size == 0:
            if return_indices:
                return list(), list()
            else:
                return list()
        else:
            overlaps = np.dot(dense_pattern, self.patterns.T)
            pattern_sizes = self.patterns.sum(axis=1)
            overlaps = overlaps / pattern_sizes
            indices = np.flatnonzero(overlaps >= self.overlap_threshold)
            options = self.patterns[indices]
            # convert to list of sparse representations
            if return_indices:
                return [np.flatnonzero(option) for option in options], indices
            else:
                return [np.flatnonzero(option) for option in options]

    def get_sparse_patterns(self):
        return [np.flatnonzero(pattern) for pattern in self.patterns]


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
    bg: BasalGanglia2
    sm: SpatialMemory

    def __init__(self,
                 tm: ApicalBasalFeedbackTM,
                 sm: SpatialMemory,
                 sp=None,
                 bg=None,
                 id_=None,
                 level=None,
                 predicted_boost=0.2,
                 min_feedback_boost=0,
                 max_feedback_boost=1,
                 gamma=0.9):
        
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

        self.predicted_columns = SDR(self.tm.basal_columns)

        self.basal_columns = tm.basal_columns
        
        self.basal_in = list()
        self.apical_in = list()
        self.feedback_in = list()

        self.basal_out = list()
        self.apical_out = list()
        self.feedback_out = list()

        self.anomaly = -1
        self.confidence = 1
        self.anomaly_threshold = 0
        self.confidence_threshold = 0

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

        self.reward = 0
        self.k = 0
        self.gamma = gamma
        self.made_decision = False
        self.current_option = None
        self.failed_option = None
        self.completed_option = None
        self.predicted_options = None

        self.predicted_boost = predicted_boost
        self.min_feedback_boost = min_feedback_boost
        self.max_feedback_boost = max_feedback_boost
        self.feedback_boost = 0

        self.learn_tm = True
        self.learn_sp = True
        self.learn_sm = True

    def __str__(self):
        return f"Block_{self.id}"

    def compute(self, add_exec=False, learn_exec=False):
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

            self.anomaly = -float('inf')
            self.confidence = float('inf')
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

            # add apical predictions, if there are no feedback predictions
            # probably, need to define a threshold
            # if self.tm.exec_predictive_cells.size == 0:
            #     self.should_return_apical_predictions = True

            self.anomaly = -float('inf')
            self.confidence = float('inf')

            # evaluate feedback boost
            self.feedback_boost = self.min_feedback_boost + total_value * (self.max_feedback_boost - self.min_feedback_boost)
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

            # form apical input sdr(cells)
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

            # form feedback input sdr(columns)
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
            # refresh patterns
                if self.learn_sm:
                    self.sm.add(self.sp_output.dense.copy())
            else:
                if self.learn_sm:
                    self.sm.add(self.sp_input.dense.copy())
            # model forgetting
            if self.learn_sm:
                self.sm.forget()
            # TM
            self.tm.set_active_columns(basal_active_columns)
            self.tm.activate_cells(self.learn_tm)

            self.anomaly = self.tm.anomaly[-1]
            self.anomaly_threshold = self.tm.anomaly_threshold

            self.tm.set_active_apical_cells(apical_active_cells)
            self.tm.set_winner_apical_cells(apical_winner_cells)
            self.tm.set_active_feedback_columns(feedback_active_columns)

            self.tm.activate_basal_dendrites()
            self.tm.activate_apical_dendrites()
            self.tm.activate_inhib_dendrites()

            self.tm.predict_cells()

            self.confidence = self.tm.confidence[-1]
            self.confidence_threshold = self.tm.confidence_threshold

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
            if self.bg is not None:
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

                if len(options) > 0:
                    boost_predicted_options = np.zeros(len(self.sm))
                    if indices.size > 0:
                        # boost predicted options
                        boost_predicted_options[indices] += self.predicted_boost
                        # feedback boost
                        boost_predicted_options[indices] += self.feedback_boost

                    option, option_value, option_values, option_index = self.bg.choose(options, condition, option_weights=boost_predicted_options,
                                                                                       return_option_value=True, return_values=True, return_index=True)
                    # reinforce good patterns and punish bad ones
                    if self.learn_sm:
                        self.sm.reinforce(option_values[0])

                    self.made_decision = True
                    self.current_option = option_index
                    self.failed_option = None
                    self.predicted_options = indices
                    # jumped off a high level option
                    if not np.isin(option_index, indices):
                        self.feedback_boost = 0
                        for block in self.feedback_in:
                            block.failed_option = block.current_option
                            k = block.k
                            if k == 0:
                                block.bg.current_option = None
                                block.bg.current_condition = None
                            else:
                                block.reinforce()
                            block.reinforce(external_value=option_value[1])

                    if return_value:
                        return option, option_value[0]
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

    def add_reward(self, reward):
        if self.made_decision and (self.bg is not None):
            self.reward += (self.gamma ** self.k) * reward
            self.k += 1

    def reinforce(self, external_value=None):
        if (self.k != 0) and self.made_decision and (self.bg is not None):
            self.bg.force_dopamine(self.reward, k=self.k)
            self.k = 0
            self.reward = 0
        elif (external_value is not None) and (self.bg is not None):
            self.bg.force_dopamine(self.reward, next_external_value=external_value)

        self.made_decision = False
        self.current_option = None

    def reset(self):
        self.tm.reset()

        if self.bg is not None:
            self.bg.reset()

        self.reward = 0
        self.k = 0
        self.made_decision = False
        self.current_option = None
        self.failed_option = None

        self.should_return_exec_predictions = False
        self.should_return_apical_predictions = False

        self.feedback_in_pattern = np.empty(0)
        self.apical_in_pattern = np.empty(0)
        self.basal_in_pattern = np.empty(0)

    def freeze(self):
        self.learn_sp = False
        self.learn_tm = False
        self.learn_sm = False
        if self.bg is not None:
            self.bg.learn_sp = False

    def unfreeze(self):
        self.learn_sp = True
        self.learn_tm = True
        self.learn_sm = True
        if self.bg is not None:
            self.bg.learn_sp = True


class InputBlock:
    """
    Dummy block for input patterns
    :param columns: int
    Number of input columns
    """
    def __init__(self, columns, id_=None, level=None):
        self.pattern = np.empty(0)

        self.basal_columns = columns
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

    def __str__(self):
        return f"InputBlock_{self.id}"

    def get_output(self, *args, **kwargs):
        return self.pattern

    def set_pattern(self, pattern):
        self.pattern = pattern

    def get_in_sizes(self):
        return 0, 0, 0

    def add_reward(self, reward):
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

    def __init__(self, blocks: list, input_blocks: list, output_block: int, block_connections: list, logs_dir=None):
        self.queue = list()
        self.blocks = blocks
        self.input_blocks = [blocks[i] for i in input_blocks]
        self.output_block = blocks[output_block]
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

        if (block.anomaly > block.anomaly_threshold) and (block.confidence >= block.confidence_threshold):
            tasks = zip(block.basal_out, [None]*len(block.basal_out))
            self.queue.append((block, {'learn_exec': True}))
            self.queue.extend(tasks)
        elif block.confidence < block.confidence_threshold:
            self.queue.append((block, {'add_exec': True}))
            # end of an option
            if block.anomaly <= block.anomaly_threshold:
                for block in block.feedback_in:
                    block.completed_option = block.current_option
                    block.reinforce()

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

    def add_rewards(self, rewards):
        for reward, block in zip(rewards, self.blocks):
            block.add_reward(reward)

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
