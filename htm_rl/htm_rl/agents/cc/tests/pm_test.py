from htm_rl.agents.cc.memory import PatternMemory
import numpy as np
from htm.bindings.sdr import SDR

EPS = 1e-12


def get_labels(pm: PatternMemory, data, input_size):
    labels = dict()
    input_pattern = SDR(input_size)
    for i, item in enumerate(data):
        input_pattern.sparse = item
        labels[i] = pm.compute(input_pattern, False)
        if labels[i] is None:
            raise ValueError('Pattern have not learned!')
    return labels


def train(pm: PatternMemory, data, epochs, input_size, noise=0):
    input_pattern = SDR(input_size)
    indices = np.arange(len(data))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            if noise > 0:
                bits_to_remove = np.random.sample(data[i], noise)
                bits_to_add = np.random.sample(np.arange(input_size), noise)
                noisy_sample = np.setdiff1d(data[i], bits_to_remove)
                noisy_sample = np.union1d(noisy_sample, bits_to_add)
            else:
                noisy_sample = data[i]
            input_pattern.sparse = noisy_sample
            pm.compute(input_pattern, True)
        print(f'epoch {epoch}: {get_labels(pm, data, input_size)}')

    labels = get_labels(pm, data, input_size)
    return labels


def test_retrieval(pm: PatternMemory, data, labels):
    iou = list()
    for label, item in zip(labels, data):
        pattern = pm.get_pattern(label)
        iou.append(np.intersect1d(pattern, item).size/(np.union1d(pattern, item).size + EPS))
    return sum(iou)/len(iou)


def generate_data(input_size, n_patterns, sparsity):
    data = [np.random.choice(np.arange(0, input_size), max(int(input_size * sparsity), 1), replace=False) for _ in range(n_patterns)]
    return data


def main():
    input_size = 1000
    epochs = 5
    seed = 5436
    n_patterns = 10
    sparsity = 0.1
    config = dict(
        input_size=1000,
        num_cells=10,
        radius=4,
        min_distance=0.05,
        permanence_increment=0.1,
        permanence_decrement=0.1,
        segment_decrement=0.1,
        permanence_connected_threshold=0.5,
        activation_threshold=1,
        activity_period=100,
        min_activity_threshold=0.001,
        boost_factor=2,
        seed=seed
    )
    data = generate_data(input_size, n_patterns, sparsity)
    pm = PatternMemory(**config)
    labels = train(pm, data, epochs, input_size)
    mean_iou = test_retrieval(pm, data, labels)

    print(mean_iou)


if __name__ == '__main__':
    main()
