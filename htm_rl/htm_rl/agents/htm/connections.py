# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Frederick C. Rotbart
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

from htm.bindings.algorithms import Connections as CPPConnections
import numpy as np


class Connections(CPPConnections):
    def filterSegmentsByCell(self, segments, cells, invert=False):
        """
        Return the subset of segments that are on the provided cells.

        @param segments
            The segments to filter. Must be sorted by cell.

        @param cells
            The cells whose segments we want to keep. Must be sorted.

        """
        mask = np.isin(self.mapSegmentsToCells(segments), cells, invert=invert)
        return segments[mask]

    def mapSegmentsToCells(self, segments):
        """
        Get the cell for each provided segment.

        @param segments
            The segments to query

        @param cells
            Output array with the same length as 'segments'
        """
        cells = super(Connections, self).mapSegmentsToCells(segments)
        return np.array(cells, dtype=np.uint32)

    def getSegmentCounts(self, cells):
        """
        Get the number of segments on each of the provided cells.

        @param cells
        The cells to check

        @param counts
        Output array with the same length as 'cells'
        """
        return np.array([self.numSegments(cell) for cell in cells], dtype=np.uint32)


if __name__ == '__main__':
    from htm.advanced.algorithms.connections import Connections as OldConnections
    co = OldConnections(numCells=1000, connectedThreshold=0.5)
    c = Connections(numCells=1000, connectedThreshold=0.5)
    import random
    for i in range(10000):
        cell = random.randint(0, 999)
        co.createSegment(99, 32)
        c.createSegment(99, 32)

    import time
    start = time.time()
    co.mapSegmentsToCells(list(range(co.numSegments())))
    end = time.time()
    print(f'old time: {end - start}')

    start = time.time()
    c.mapSegmentsToCells(list(range(c.numSegments())))
    end = time.time()
    print(f'new time: {end - start}')
