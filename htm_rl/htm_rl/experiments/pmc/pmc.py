import kohonen
import numpy as np


class PMCToM1Basic:
    def __init__(self,
                 input_size: int,
                 limits: list[tuple[float, float]],
                 n_neurons: int,
                 learning_rate: float = 1,
                 neighbourhood_radius: float = 0.1,
                 noise: float = 0.1,
                 allow_add_new_neurons: bool = False,
                 seed=None,
                 ):
        self.input_size = input_size
        self.limits = limits
        self.n_neurons = n_neurons
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        assert len(limits) == input_size

        neighbourhood_radius *= n_neurons
        CT = kohonen.ConstantTimeseries

        kw = dict(
            shape=(n_neurons,),
            dimension=input_size,
            learning_rate=CT(learning_rate),
            noise_variance=noise,
            neighborhood_size=neighbourhood_radius,
            seed=seed
        )
        if allow_add_new_neurons:
            self.pmc = kohonen.GrowingGas(
                growth_interval=7, max_connection_age=17, **kw)
        else:
            self.pmc = kohonen.Gas(**kw)

        self.pmc.reset(self.initialize_neuron)

    def compute(self, sparse_pattern, weights=None):
        chosen_neurons = self.pmc.neurons[sparse_pattern, :].squeeze()
        if weights is None:
            weights = np.ones(sparse_pattern.size)

        weights = weights.reshape((weights.size, 1))
        norm = np.sum(weights)

        if norm != 0:
            value = np.sum(chosen_neurons * weights, axis=0)/norm
        else:
            value = np.zeros(self.input_size)

        self.pmc.learn(value)

        return value

    def initialize_neuron(self, neuron_id):
        low = [x[0] for x in self.limits]
        high = [x[1] for x in self.limits]
        return self.rng.uniform(low=low, high=high, size=self.input_size)


if __name__ == '__main__':
    pmc = PMCToM1Basic(
        n_neurons=100,
        input_size=7,
        limits=[(0, 1)]*7
    )

    print(pmc.compute([np.random.randint(0, 100, size=10)]))
