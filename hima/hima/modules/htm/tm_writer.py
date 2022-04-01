# 1. Active cells
# 2. Predictive cells
# 3. Winner cells
# 4. Segments for cell
# 5. Synapses for segment: Presynaptic cell and permanence
from hima.modules.htm.temporal_memory import ApicalBasalFeedbackTM
from pickle import dump
import os


class HTMWriter:
    def __init__(self, name, directory, tm: ApicalBasalFeedbackTM, save_every=None):
        self.directory = directory
        self.name = name
        self.time_step = 0
        self.save_every = save_every
        self.cells = list()
        self.basal_symbols = list()
        self.apical_symbols = list()
        self.feedback_symbols = list()

        self.tm = tm
        self.info = {'basal_range': tm.basal_range,
                     'apical_range': tm.apical_range,
                     'feedback_range': tm.feedback_range,
                     'basal_columns': tm.basal_columns,
                     'basal_cells_per_column': tm.basal_cells_per_column,
                     'apical_columns': tm.apical_columns,
                     'apical_cells_per_column': tm.apical_cells_per_column,
                     'feedback_columns': tm.feedback_columns,
                     'feedback_cells_per_columns': 1}
        with open(os.path.join(directory, name + '_info.pkl'), 'wb') as file:
            dump(self.info, file)

    def write(self, basal_symbol=None, apical_symbol=None, feedback_symbol=None, save=False):
        cells = list()
        for cell_id in range(self.tm.total_cells):
            basal_segments = self.tm.basal_connections.segmentsForCell(cell_id)
            apical_segments = self.tm.apical_connections.segmentsForCell(cell_id)
            inhibit_segments = self.tm.inhib_connections.segmentsForCell(cell_id)
            exec_segments = self.tm.exec_feedback_connections.segmentsForCell(cell_id)
            if self.tm.basal_range[0] <= cell_id < self.tm.basal_range[1]:
                type_ = 0  # basal
                active = cell_id in self.tm.active_basal_cells.sparse
                winner = cell_id in self.tm.winner_basal_cells.sparse
            elif self.tm.apical_range[0] <= cell_id < self.tm.apical_range[1]:
                type_ = 1  # apical
                active = cell_id in self.tm.active_apical_cells.sparse
                winner = cell_id in self.tm.winner_apical_cells.sparse
            elif self.tm.feedback_range[0] <= cell_id < self.tm.feedback_range[1]:
                type_ = 2  # feedback
                active = cell_id in self.tm.active_feedback_columns.sparse
                winner = False
            else:
                raise ValueError("Cell id is out of range")

            predictive = cell_id in self.tm.predicted_cells.sparse

            cells.append({'segments': {'basal': self.write_segments(basal_segments, self.tm.active_basal_segments,
                                                                    self.tm.matching_basal_segments,
                                                                    self.tm.basal_connections,
                                                                    self.tm.connected_threshold),
                                       'apical': self.write_segments(apical_segments, self.tm.active_apical_segments,
                                                                     self.tm.matching_apical_segments,
                                                                     self.tm.apical_connections,
                                                                     self.tm.connected_threshold),
                                       'inhib': self.write_segments(inhibit_segments, self.tm.active_inhib_segments,
                                                                    self.tm.matching_inhib_segments,
                                                                    self.tm.inhib_connections,
                                                                    self.tm.connected_threshold),
                                       'exec': self.write_segments(exec_segments, self.tm.active_exec_segments,
                                                                   self.tm.matching_exec_segments,
                                                                   self.tm.exec_feedback_connections,
                                                                   self.tm.connected_threshold)},
                          'type': type_,
                          'active': active,
                          'winner': winner,
                          'predictive': predictive,
                          'id': cell_id})
        self.cells.append(cells)
        self.basal_symbols.append(basal_symbol)
        self.apical_symbols.append(apical_symbol)
        self.feedback_symbols.append(feedback_symbol)

        self.time_step += 1

        if save:
            self.save()
        else:
            if self.save_every is not None:
                if (self.time_step % self.save_every) == 0:
                    self.save()

    def save(self):
        with open(os.path.join(self.directory, self.name + f'_{self.time_step}.pkl'), 'wb') as file:
            dump({'cells': self.cells,
                  'symbols': {'basal': self.basal_symbols,
                              'apical': self.apical_symbols,
                              'feedback': self.feedback_symbols}}, file)

        self.cells.clear()
        self.basal_symbols.clear()
        self.apical_symbols.clear()
        self.feedback_symbols.clear()

    @staticmethod
    def write_segments(segment_ids, active_segments, matching_segments, connections, permanence_threshold):
        segments = list()
        for segment in segment_ids:
            synapses = connections.synapsesForSegment(segment)
            synapses = [(s,
                         connections.presynapticCellForSynapse(s),
                         connections.permanenceForSynapse(s),
                         connections.permanenceForSynapse(s) >= permanence_threshold) for s in synapses]
            if segment in active_segments:
                state = 1  # active
            elif segment in matching_segments:
                state = 2  # matching
            else:
                state = 0  # inactive

            segments.append({'synapses': synapses,
                             'state': state,
                             'id': segment})
        return segments
