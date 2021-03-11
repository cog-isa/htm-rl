import numpy as np
from htm.bindings.sdr import SDR
from htm_apical_basal_feeedback import ApicalBasalFeedbackTM
import os
import pickle


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
    def __init__(self,
                 tm: ApicalBasalFeedbackTM,
                 sp=None,
                 bg=None,
                 id_=None,
                 level=None,
                 pattern_overlap_threshold=0.9):
        
        self.tm = tm
        self.sp = sp
        self.bg = bg
        self.patterns = Patterns(pattern_overlap_threshold, self.tm.activation_threshold)

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

    def __str__(self):
        return f"Block_{self.id}"

    def compute(self, learn=True, add_exec=False, learn_exec=False):
        self.should_return_exec_predictions = False
        self.should_return_apical_predictions = False
        # gather all inputs
        # form basal input sdr(columns)
        if learn and learn_exec:
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
            self.tm.learn_exec_feedback_segments()

            self.feedback_in_pattern = feedback_active_columns
            self.apical_in_pattern = np.empty(0)
            self.basal_in_pattern = np.empty(0)

            self.anomaly = -float('inf')
            self.confidence = float('inf')
        elif add_exec:
            self.should_return_exec_predictions = True
            feedback_active_columns = list()
            shift = 0

            for block in self.feedback_in:
                feedback_active_columns.append(block.get_output('feedback') + shift)
                shift += block.basal_columns

            if len(feedback_active_columns) > 0:
                feedback_active_columns = np.concatenate(feedback_active_columns)
            else:
                feedback_active_columns = np.empty(0)

            self.tm.set_active_feedback_columns(feedback_active_columns)
            self.tm.activate_exec_dendrites()

            self.feedback_in_pattern = feedback_active_columns

            # add apical predictions, if there are no feedback predictions
            # probably, need to define a threshold
            if self.tm.exec_predictive_cells.size == 0:
                self.should_return_apical_predictions = True

            self.anomaly = -float('inf')
            self.confidence = float('inf')
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
                self.sp.compute(self.sp_input, learn, self.sp_output)
                basal_active_columns = self.sp_output.sparse
            # refresh patterns
                self.patterns.add(self.sp_output.dense)
            else:
                self.patterns.add(self.sp_input.dense)
            # TM
            self.tm.set_active_columns(basal_active_columns)
            self.tm.activate_cells(learn)

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

    def get_output(self, mode):
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

            apical_input = SDR(shift)
            apical_input.sparse = apical_active_columns
            # filter columns by Basal Ganglia conditioned on apical input
            if self.bg is not None:
                return self.bg.choose(self.patterns.get_options(self.predicted_columns.dense), apical_input)
            else:
                return predicted_columns
        else:
            raise ValueError(f'There is no such mode {mode}!')

    def get_in_sizes(self):
        self.feedback_in_size = sum([block.basal_columns for block in self.feedback_in])
        self.apical_in_size = sum([block.tm.basal_total_cells for block in self.apical_in])
        self.basal_in_size = sum([block.basal_columns for block in self.basal_in])
        return self.feedback_in_size, self.apical_in_size, self.basal_in_size


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

    def __str__(self):
        return f"InputBlock_{self.id}"

    def get_output(self, *args, **kwargs):
        return self.pattern

    def set_pattern(self, pattern):
        self.pattern = pattern

    def get_in_sizes(self):
        return 0, 0, 0


class Hierarchy:
    """
    Compute hierarchy of blocks
    :param blocks: list(Block)
    :param input_blocks: list(InputBlocks)
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
    def __init__(self, blocks: list, input_blocks: list, block_connections: list, logs_dir=None):
        self.queue = list()
        self.blocks = blocks
        self.input_blocks = input_blocks
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

    def compute(self, learn=True):
        block, kwargs = self.queue.pop()

        if kwargs is None:
            block.compute(learn=learn)
        else:
            block.compute(**kwargs)

        if (block.anomaly > block.anomaly_threshold) and (block.confidence >= block.confidence_threshold):
            tasks = zip(block.basal_out, [{'learn': learn}]*len(block.basal_out))
            self.queue.append((block, {'learn': learn, 'learn_exec': True}))
            self.queue.extend(tasks)
        elif (block.anomaly <= block.anomaly_threshold) and (block.confidence < block.confidence_threshold):
            self.queue.append((block, {'learn': learn, 'add_exec': True}))

        # logging
        if self.logs_dir is not None:
            self.logs['tasks'].append((block.id, kwargs))
            self.logs['patterns']['basal_out'].append(block.get_output('basal'))
            self.logs['patterns']['feedback_in'].append(block.feedback_in_pattern)
            self.logs['anomaly'].append((block.anomaly, block.anomaly_threshold))
            self.logs['confidence'].append((block.confidence, block.confidence_threshold))

        if len(self.queue) > 0:
            self.compute(learn=learn)

    def set_input(self, patterns, learn=True):
        """
        Set patterns to input blocks
        :param patterns: list of lists of active bits or None
        None if there is no pattern for specific input
        For example: [None, [1, 2, 3], [4, 5, 6]]
        So there is no pattern for the first input block
        :param learn: bool
        If true, then memory will learn patterns
        :return:
        """
        for pattern, block in zip(patterns, self.input_blocks):
            # logging
            if self.logs_dir is not None:
                self.logs['tasks'].append((block.id, {'learn': learn, 'input_block': True}))
                self.logs['patterns']['basal_out'].append(pattern)
                self.logs['patterns']['feedback_in'].append(np.empty(0))
                self.logs['anomaly'].append(None)
                self.logs['confidence'].append(None)

            if pattern is not None:
                block.set_pattern(pattern)
                tasks = list(zip(block.basal_out, [{'learn': learn}]*len(block.basal_out)))
                self.queue.extend(tasks)
        self.queue = self.queue[::-1]
        # start processing input
        self.compute(learn=learn)

    def save_logs(self):
        if self.logs_dir is not None:
            with open(os.path.join(self.logs_dir, 'logs.pkl'), 'wb') as file:
                pickle.dump(self.logs,
                            file)
        else:
            raise ValueError('Log dir is not defined!')


class Patterns:
    def __init__(self, overlap_threshold: float, pattern_size: int):
        self.patterns = np.empty(0)
        self.pattern_size = pattern_size
        self.overlap_threshold = int(pattern_size * overlap_threshold)

    def add(self, dense_pattern: np.array):
        if self.patterns.size == 0:
            self.patterns = dense_pattern.reshape((1, -1))
        else:
            overlaps = np.dot(dense_pattern, self.patterns.T)
            if np.any(overlaps > self.overlap_threshold) > 0:
                best_index = np.argmax(overlaps)
                self.patterns[best_index] = dense_pattern
            else:
                self.patterns = np.vstack(self.patterns, dense_pattern.reshape(1, -1))

    def get_options(self, dense_pattern):
        if self.patterns.size == 0:
            return np.empty(0)
        else:
            overlaps = np.dot(dense_pattern, self.patterns.T)
            options = self.patterns[overlaps > self.overlap_threshold] * dense_pattern
            # convert to list of sparse representations
            return [np.nonzero(option) for option in options]
