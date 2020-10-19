import json

class Writer():
    def __init__(self, tm):
        self.tm = tm
        self.step = 0
        columns, cells_per_column = tm.getColumnDimensions()[0], tm.getCellsPerColumn()
        self.num_cells = columns * cells_per_column
        params = {'cells_per_column': cells_per_column,
                  'num_cells': self.num_cells}
        self.data = {'params': params}

    def write(self):
        active_cells = self.tm.getActiveCells().sparse
        predictive_cells = self.tm.getPredictiveCells().sparse
        active_segments = self.tm.getActiveSegments()

        cells = {}
        for cell in range(self.num_cells):
            cell_dict = {}
            cell_dict['St'] = 0
            if cell in active_cells: cell_dict['St'] = 1
            if cell in predictive_cells: cell_dict['St'] = 2
            if cell in active_cells and cell in predictive_cells: cell_dict['St'] = 3
            segments = self.tm.connections.segmentsForCell(cell)
            dict_segments = {}
            for segment in segments:
                dict_segment = {}
                dict_segment['St'] = 1 if segment in active_segments else 0
                dict_segment['Ce'] = [self.tm.connections.presynapticCellForSynapse(synapse) for synapse
                                      in self.tm.connections.synapsesForSegment(segment)]
                dict_segments[segment] = dict_segment
            cell_dict['Se'] = dict_segments
            cells[cell] = cell_dict

        self.data[self.step] = cells
        self.step = self.step + 1

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump(self.data, outfile, indent=1)