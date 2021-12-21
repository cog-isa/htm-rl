import numpy as np
from utils import softmax, bsu
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from PIL import Image


class ThaPMCToM1:
    def __init__(self,
                 input_size: int,
                 n_neurons: int,
                 learning_rate: float = 1,
                 sparsity: float = 0.01,
                 permanence_increment: float = 0.1,
                 permanence_decrement: float = 0.01,
                 connected_threshold: float = 0.5,
                 initial_permanence: float = 0.4,
                 softmax_beta: float = 1.0,
                 bsu_k: float = 1,
                 seed=None
                 ):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.k_top = int(self.sparsity * self.n_neurons)
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.connected_threshold = connected_threshold
        self.initial_permanence = initial_permanence
        self.softmax_beta = softmax_beta
        self.bsu_k = bsu_k
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # initialize neurons
        self.neurons = self.rng.uniform(size=(self.n_neurons, self.input_size))
        self.specializations = self.rng.uniform(size=(self.n_neurons, self.input_size))
        self.connections = np.zeros((self.n_neurons, self.n_neurons)) + self.initial_permanence
        np.fill_diagonal(self.connections, 1)

    def compute(self, bg_modulation, learn=True):
        # produce output
        # choose cluster center
        connected = self.connections > self.connected_threshold
        scores = np.dot(connected, bg_modulation)
        cluster_center_probs = softmax(self.softmax_beta*scores)
        cluster_center = self.rng.choice(len(scores), 1, p=cluster_center_probs)[0]
        # recruit additional cells from connection pool
        pool = np.flatnonzero(connected[cluster_center])
        pool_probs = bsu(bg_modulation[pool], k=self.bsu_k)
        cells = pool[self.rng.uniform(size=pool_probs.size) < pool_probs]
        cells = np.union1d(cells, cluster_center)
        # calculate output
        out = np.sum(self.neurons[cells] * self.specializations[cells], axis=0) / np.sum(self.specializations[cells], axis=0)
        # learn
        if learn:
            self.learn(out)

        return out, cells

    def learn(self, out):
        distance = self.cue_distance(out)
        k_top = np.argpartition(distance, kth=-self.k_top)[-self.k_top:]
        # shift receptive field
        deltas = self.learning_rate * self.specializations[k_top] * (out - self.neurons[k_top])
        self.neurons[k_top] += deltas
        # adjust connections
        k_top_connections = self.connections[k_top]
        k_top_connected = k_top_connections > self.connected_threshold

        connections_to_increase = np.zeros_like(k_top_connections, dtype=bool)
        connections_to_increase[:, k_top] = True
        # TODO should we make it reciprocal?
        connections_to_decrease = k_top_connected & ~connections_to_increase

        k_top_connections[connections_to_increase] += self.permanence_increment
        k_top_connections[connections_to_decrease] -= self.permanence_decrement

        k_top_connections = np.clip(k_top_connections, 0, 1)

        self.connections[k_top] = k_top_connections

    def cue_distance(self, cue):
        distance = np.linalg.norm(np.sqrt(self.specializations)*(cue - self.neurons),
                                  axis=1)
        return distance

    def distance_matrix_heatmap(self):
        dmat = distance_matrix(self.neurons,
                               self.neurons)
        avdist = dmat.mean(axis=0)
        image = np.zeros(int(round(avdist.size ** 0.5)) ** 2)
        image[:avdist.size] = avdist
        image = image.reshape((int(image.size ** 0.5), -1))
        image /= avdist.max()
        cm = plt.get_cmap()
        colored = cm(image)
        return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


if __name__ == '__main__':
    pmc = ThaPMCToM1(3, 10, sparsity=0.3, seed=321)
    for i in range(100):
        x = pmc.compute(np.zeros(10))

    print(pmc.connections)
