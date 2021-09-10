import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import SpatialPooler
from itertools import product
from copy import deepcopy

EPS = 1e-12


class Memory:
    """
    The Memory object saves SDR representations of states and clusterizes them using the similarity measure.
    The SDR representation must have fixed sparsity of active cells for correct working.

    Parameters
    ----------
    size : int
        The size is the size of SDR representations, which are stored
    threshold: float
        The threshold is used to determine then it's necessary to create a new cluster.

    Attributes
    ----------
    size: int
        It stores size argument.
    kernels : np.array
        This is the list of created clusters representations in dence form. It contains information about frequency of
        cell's activity (for each cluster) during working. Its shape: (number of clusters, size).
    norms: np.array
        This is the list of representations amount for each cluster. Its shape: (munber of clusters, 1)
    threshold: float
        It stores threshold argument.
    """

    def __init__(self, size, threshold=0.5):
        self.kernels = None
        self.norms = None
        self.threshold = threshold
        self.size = size

    def add(self, state):
        """ Add a new SDR representation (store and clusterize).

        Parameters
        ----------
        state: np.array
            This is the SDR representation (sparse), that we want to store ande clusterize with other stored SDRs.

        Returns
        -------
        """
        state_dense = np.zeros(self.size)
        state_dense[state] = 1
        sims = self.similarity(state_dense)
        if np.sum(sims > self.threshold) == 0:
            if self.kernels is None:
                self.kernels = state_dense.reshape((1, -1))
                self.norms = np.array([[1]])
            else:
                self.kernels = np.vstack((self.kernels, state_dense))
                self.norms = np.vstack((self.norms, [1]))
        else:
            self.kernels[np.argmax(sims)] += state_dense
            self.norms[np.argmax(sims)] += 1

    def similarity(self, state):
        """This function evaluate similarity measure between stored clusters and new state.

        Parameters
        ----------
        state: np.array
            The sparse representation of the state to be compared.

        Returns
        -------
        similarities: np.array
            The similarity measures for given state. If the Memory object don't have any saved clusters, then the empty
            array is returned, else returned array contained similarities between the state and each cluster.
            Its shape: (number of kernels, 1).

        """
        if self.kernels is None:
            return np.array([])
        else:
            normalised_kernels = self.kernels / self.norms
            sims = normalised_kernels @ state.T / (
                    np.sqrt(np.sum(normalised_kernels ** 2, axis=1)) * np.sqrt(state @ state.T))
            similarities = sims.T
            return similarities

    def adopted_kernels(self, sparsity):
        """This function normalises stored representations and cuts them by sparsity threshold.

        Parameters
        ----------
        sparsity: float
            The sparsity of active cells in stored SDR representations.

        Returns
        -------
        clusters_representations: np.array
            Normalised and cutted representations of each cluster. The cutting is done by choosing the most frequent
            active cells (their number is defined by sparsity) in kernels attribute. All elements of array are
            in [0, 1]. The shape is (number of clusters, 1).
        """
        data = np.copy(self.kernels)
        data[data < np.quantile(data, 1 - sparsity, axis=1).reshape((-1, 1))] = 0
        clusters_representations = data / self.norms
        return clusters_representations


class Empowerment:
    """
    The Empowerment object contains all necessary things to evaluate 'empowerment' using the model of environment. This
    model creates and learns also here.

    Parameters
    ----------
    seed: int
        The seed for random generator.
    encode_size: int
        The size of SDR representations which is taken by model.
    tm_config: dict
        It contains all parameters for initialisation of the TemporalMemory without the columnDimensions.
        columnDimensions is defined inside Empowerment.
    sparsity: float
        The sparsity of SDR representations which are used in the TemporalMemory algorithm.
    sp_config (optional): dict
        It contains all parameters for initialisation of the SpatialPooler without the inputDimensions
        and localareaDensity. They are defined inside Empowerment. By default sp_config is None that means the absence
        of SpatialPooler.
    memory (optional): bool
        This parameter defines will be used the Memory for saving and clusterization of state representations or not.
        By default is False (doesn't use the Memory).
    similarity_threshold (optional): float
        This parameter determines the threshold for cluster creation. It is used then memory is True. By default: 0.6.
    evaluate (optional): bool
        This flag defines the necessarity of storing some statistics to evaluate the learning process.
        By default is True.

    Attributes
    ----------
    evaluate: bool
        It stores the same parameter.
    anomalies: list
        It stores the anomaly values of TM for easch time step after learning. Only then evaluate is True.
    IoU: list
        It stores the Intersection over Union values of TM predictions and real ones for easch time step after learning.
        Only then evaluate is True.
    sparsity: float
        It stores the same parameter.
    sp: SpatialPooler
        It contains the SpatialPooler object if it was defined, else None
    tm: TemporalMemory
        It contains the TemporalMemory object.
    size: int
        It stores the encode_size parameter.
    memory: Memory
        It contains the Memory object if memory parameter is True, else None.
    """

    def __init__(self, seed, encode_size, tm_config, sparsity,
                 sp_config=None,
                 memory=False,
                 similarity_threshold=0.6,
                 evaluate=True):
        self.evaluate = evaluate
        if evaluate:
            self.anomalies = []
            self.IoU = []
        self.sdr_0 = SDR(encode_size)
        self.sdr_1 = SDR(encode_size)
        self.sparsity = sparsity

        if sp_config is not None:
            self.sp = SpatialPooler(inputDimensions=[encode_size],
                                    seed=seed,
                                    localAreaDensity=sparsity,
                                    **sp_config,
                                    )
            self.tm = TemporalMemory(
                columnDimensions=self.sp.getColumnDimensions(),
                seed=seed,
                **tm_config,
            )
            self.sdr_sp = SDR(self.sp.getColumnDimensions())
            self.size = self.sp.getColumnDimensions()[0]
        else:
            self.sp = None
            self.tm = TemporalMemory(
                columnDimensions=[encode_size],
                seed=seed,
                **tm_config,
            )
            self.size = self.tm.getColumnDimensions()[0]

        if memory:
            self.memory = Memory(self.tm.getColumnDimensions()[0], threshold=similarity_threshold)
        else:
            self.memory = None

    def eval_state(self, state, horizon, use_segments=False, use_memory=False):
        """This function evaluates empowerment for given state.

        Parameters
        ----------
        state: np.array
            The SDR representation (sparse) of the state.
        horizon: int
            The horison of evaluating for given state. The good value is 3.
        use_segments (optional): bool
            The flag determines using of segments instead of cells to evaluate empowerment. By default: False.
        use_memory (optional): bool
            The flag determines using of the Memory object. Useful only if this object was initialised.
            By default: False

        Returns
        -------
        empowerment: float
            The empowerment value (always > 0).
        p: np.array
            The array of probabilities on that the empowerment was calculated.
        start_state: np.array
            The SDR representation of given state that is used in TM. (Only if sp is defined it differs from parameter
            state).
        """
        if self.sp is not None:
            self.sdr_0.sparse = state
            self.sp.compute(self.sdr_0, learn=False, output=self.sdr_sp)
            sdr = self.sdr_sp
        else:
            self.sdr_0.sparse = state
            sdr = self.sdr_0
        start_state = np.copy(sdr.sparse)
        data = np.zeros(self.tm.getColumnDimensions()[0])
        for actions in range(horizon):
            self.tm.reset()
            self.tm.compute(sdr, learn=False)
            self.tm.activateDendrites(learn=False)
            predictiveCells = self.tm.getPredictiveCells().sparse
            predictedColumnIndices = [self.tm.columnForCell(i) for i in predictiveCells]
            #         if len(predictedColumnIndices) == 0:
            #             print('!!!!!!')
            sdr.sparse = np.unique(predictedColumnIndices)
        if use_segments:
            predictedColumnIndices = map(self.tm.columnForCell,
                                         map(self.tm.connections.cellForSegment, self.tm.getActiveSegments()))
        for i in predictedColumnIndices:
            data[i] += 1
        if self.memory is not None and use_memory:
            if (self.memory.kernels is not None) and (self.memory.kernels.size > 0):
                p = np.round(self.memory.adopted_kernels(self.sparsity) @ data.T / (self.sparsity * self.size))
                total_p = p.sum()
                empowerment = np.sum(-p / (total_p + EPS) * np.log(p / (total_p + EPS), where=p != 0), where=p != 0)
                p = p / (total_p + EPS)
                return empowerment, p, start_state
            else:
                return 0, None, start_state
        empowerment = np.sum(-data / data.sum() * np.log(data / data.sum(), where=data != 0), where=data != 0)
        p = data / data.sum()
        return empowerment, p, start_state

    def eval_env(self, environment, horizon, use_segments=False, use_memory=False):
        """This function evaluate empowerment for every state in gridworld environment.

        Parameters
        ----------
        environment:
            The gridworld environment to be evaluated.
        horizon: int
            The horison of evaluating for given state. The good value is 3.
        use_segments (optional): bool
            The flag determines using of segments instead of cells to evaluate empowerment. By default: False.
        use_memory (optional): bool
            The flag determines using of the Memory object. Useful only if this object was initialised.
            By default: False

        Returns
        -------
        empowerment_map: np.array
            This is the map of the environment with values of empowerment for each state.
        """
        env = deepcopy(environment)
        empowerment_map = np.zeros(env.env.shape)

        for i in range(env.env.shape[0]):
            for j in range(env.env.shape[1]):
                if not env.env.entities['obstacle'].mask[i, j]:
                    env.env.agent.position = (i, j)
                    _, s, _ = env.observe()
                    empowerment_map[i, j] = self.eval_state(s, horizon, use_segments, use_memory)[0]
        # plt.imshow(empowerment_map)
        # plt.show()
        return empowerment_map

    def learn(self, state_0, state_1):
        """This function realize learning of TM.

        Parameters
        ----------
        state_0: np.array
            The SDR representation of the state (sparse form).
        state_1: np.array
            The SDR representation of the next state (sparse form).

        Returns
        -------
        """
        self.sdr_0.sparse = state_0
        self.sdr_1.sparse = state_1
        self.tm.reset()

        if self.sp is not None:
            self.sp.compute(self.sdr_0, learn=True, output=self.sdr_sp)
            if self.memory is not None:
                self.memory.add(self.sdr_sp.sparse)
            self.tm.compute(self.sdr_sp, learn=True)
        else:
            if self.memory is not None:
                self.memory.add(self.sdr_0.sparse)
            self.tm.compute(self.sdr_0, learn=True)

        if self.evaluate:
            self.tm.activateDendrites(learn=False)
            predictiveCells = self.tm.getPredictiveCells().sparse
            predictedColumnIndices = np.unique([self.tm.columnForCell(i) for i in predictiveCells])

        if self.sp is not None:
            self.sp.compute(self.sdr_1, learn=True, output=self.sdr_sp)
            self.tm.compute(self.sdr_sp, learn=True)
            if self.evaluate:
                intersection = np.intersect1d(self.sdr_sp.sparse, predictedColumnIndices)
                union = np.union1d(self.sdr_sp.sparse, predictedColumnIndices)
        else:
            self.tm.compute(self.sdr_1, learn=True)
            if self.evaluate:
                intersection = np.intersect1d(self.sdr_1.sparse, predictedColumnIndices)
                union = np.union1d(self.sdr_1.sparse, predictedColumnIndices)

        if self.evaluate:
            self.IoU.append(len(intersection) / len(union))
            self.anomalies.append(self.tm.anomaly)
        self.tm.reset()

    def detailed_evaluate(self, env, horizon, use_segments=False, use_memory=False):
        """This function evaluate TM and real empowerment and confusion matrix for every state in gridworld environment.

        Parameters
        ----------
        env:
            The gridworld environment to be evaluated.
        horizon: int
            The horison of evaluating for given state. The good value is 3.
        use_segments (optional): bool
            The flag determines using of segments instead of cells to evaluate empowerment. By default: False.
        use_memory (optional): bool
            The flag determines using of the Memory object. Useful only if this object was initialised.
            By default: False

        Returns
        -------
        plot normalised maps with TM and real empowerment. Also plot confusion matrix in map style.
        """
        confusion_data = np.zeros((env.env.shape[0] * env.env.shape[1], self.tm.getColumnDimensions()[0]))
        empowerment_map = np.zeros(env.env.shape)
        real_empowerment_map = np.zeros(env.env.shape)

        for i in trange(env.env.shape[0]):
            for j in range(env.env.shape[1]):
                if not env.env.entities['obstacle'].mask[i, j]:
                    env.env.agent.position = (i, j)
                    _, s, _ = env.observe()
                    emp, _, s = self.eval_state(s, horizon, use_segments, use_memory)
                    empowerment_map[i, j] = emp
                    confusion_data[env.env.shape[1] * i + j, s] = 1
                    real_empowerment_map[i, j] = real_empowerment(env, (i, j), horizon)[0]

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        mask = empowerment_map != 0
        empowerment_map[mask] = (empowerment_map[mask != 0] - np.min(empowerment_map[mask])) / (
                np.max(empowerment_map) - np.min(empowerment_map[mask]))
        plt.imshow(empowerment_map)
        plt.colorbar()
        plt.title('TM')

        plt.subplot(122)
        mask = real_empowerment_map != 0
        real_empowerment_map[mask] = (real_empowerment_map[mask != 0] - np.min(real_empowerment_map[mask])) / (
                np.max(real_empowerment_map) - np.min(real_empowerment_map[mask]))
        plt.imshow(real_empowerment_map)
        plt.colorbar()
        plt.title('Real')
        plt.show()

        intersection = confusion_data @ confusion_data.T
        inv_mat = ~confusion_data.astype(bool)
        union = inv_mat.shape[1] - inv_mat.astype(float) @ inv_mat.astype(float).T

        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
        plot_data = iou.reshape(env.env.shape[0], env.env.shape[1], env.env.shape[0], env.env.shape[1])
        image = np.zeros((env.env.shape[0] ** 2, env.env.shape[0] ** 2))
        for i in range(env.env.shape[0]):
            for j in range(env.env.shape[1]):
                image[env.env.shape[0] * i:env.env.shape[0] * (i + 1),
                env.env.shape[1] * j:env.env.shape[1] * (j + 1)] = \
                    plot_data[i, j]

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.yticks([-0.5 + env.env.shape[0] * i for i in range(env.env.shape[0])])
        plt.xticks([-0.5 + env.env.shape[1] * i for i in range(env.env.shape[0])])
        plt.grid(linewidth=3)
        plt.colorbar()
        plt.show()


def draw_tm(tm, grid_step):
    tm.activateDendrites(learn=False)
    activeCells = tm.getActiveCells().dense
    predictedCells = tm.getPredictiveCells().dense
    data = np.zeros((tm.getColumnDimensions()[0], tm.getCellsPerColumn(), 3))
    data[:, :, 0] = activeCells
    data[:, :, 1] = predictedCells

    plt.figure(figsize=(tm.getColumnDimensions()[0] / 10, tm.getCellsPerColumn() * 2))
    plt.imshow(np.moveaxis(data, [0, 1, 2], [1, 0, 2]), aspect='auto')
    plt.yticks([-0.5 + i for i in range(tm.getCellsPerColumn())])
    plt.xticks([-0.5 + i * grid_step for i in range(tm.getColumnDimensions()[0] // grid_step)])
    plt.grid(linewidth=2)
    plt.show()


def draw_segments(tm):
    data = np.zeros(tm.getCellsPerColumn() * tm.getColumnDimensions()[0])
    max_seg = 0
    for cell in trange(tm.getCellsPerColumn() * tm.getColumnDimensions()[0]):
        segs = tm.connections.segmentsForCell(cell)
        data[cell] = len(segs)
        if len(segs) > max_seg:
            max_seg = len(segs)
    plt.figure(figsize=(tm.getColumnDimensions()[0] / 10, tm.getCellsPerColumn() * 2))
    print(f'Number of segments. Max: {max_seg}')
    plt.imshow(data.reshape((tm.getCellsPerColumn(), tm.getColumnDimensions()[0]), order='F'), aspect='auto')
    plt.show()


def draw_active_segments(tm):
    data = np.zeros(tm.getCellsPerColumn() * tm.getColumnDimensions()[0])
    for seg in tm.getActiveSegments():
        cell = tm.connections.cellForSegment(seg)
        data[cell] += 1

    plt.figure(figsize=(tm.getColumnDimensions()[0] / 10, tm.getCellsPerColumn() * 2))
    print(f'Number of segments. Max: {data.max()}')
    plt.imshow(data.reshape((tm.getCellsPerColumn(), tm.getColumnDimensions()[0]), order='F'), aspect='auto')
    plt.show()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def real_empowerment(env, position, horizon):
    data = np.zeros(env.env.shape)

    for actions in product(range(4), repeat=horizon):
        env.env.agent.position = position
        for a in actions:
            env.act(a)

        data[env.env.agent.position] += 1
    return np.sum(-data / data.sum() * np.log(data / data.sum(), where=data != 0), where=data != 0), data


def learn(seed,
          empowerment,
          env,
          steps,
          dump_step=None,
          horizon=3,
          use_segments=False,
          use_memory=False,
          ):
    np.random.seed(seed)
    visit_map = np.zeros(env.env.shape)
    encode_sizes = []

    for t in trange(steps):
        visit_map[env.env.agent.position] += 1

        a = np.random.randint(env.n_actions)
        _, s0, _ = env.observe()
        encode_sizes.append(len(s0))
        env.act(a)
        _, s1, _ = env.observe()

        empowerment.learn(s0, s1)

        if dump_step is not None:
            if (t + 1) % dump_step == 0:
                empowerment.eval_env(env, horizon, use_segments, use_memory)

    plt.title('Visit')
    plt.imshow(visit_map)
    plt.colorbar()
    plt.show()

    plt.plot(moving_average(empowerment.anomalies, 100))
    plt.title('Anomaly')
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

    plt.plot(moving_average(empowerment.IoU, 100))
    plt.title('Intersection over union')
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

    plt.plot(moving_average(encode_sizes, 100))
    plt.title('Number of active columns')
    plt.show()
