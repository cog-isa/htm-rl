from htm_rl.agents.cc.memory import PatternMemory
import numpy as np
from htm.bindings.sdr import SDR
from tqdm import tqdm

EPS = 1e-12


def get_labels(pm: PatternMemory, data, input_size):
    labels = dict()
    input_pattern = SDR(input_size)
    for i, item in enumerate(data):
        input_pattern.sparse = item
        labels[i] = pm.compute(input_pattern, False)
    return labels


def train(pm: PatternMemory, data, epochs, input_size, noise=0.0):
    input_pattern = SDR(input_size)
    indices = np.arange(len(data))
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(indices)
        for i in indices:
            if noise > 0:
                n_bits = int(noise * len(data[i]))
                bits_to_remove = np.random.choice(data[i], n_bits, replace=False)
                bits_to_add = np.random.choice(np.arange(input_size), n_bits, replace=False)
                noisy_sample = np.setdiff1d(data[i], bits_to_remove)
                noisy_sample = np.union1d(noisy_sample, bits_to_add)
            else:
                noisy_sample = data[i]
            input_pattern.sparse = noisy_sample
            pm.compute(input_pattern, True)
        # print(f'epoch {epoch}: {get_labels(pm, data, input_size)}')

    labels = get_labels(pm, data, input_size)
    return labels


def test_retrieval(pm: PatternMemory, data, labels):
    iou = list()
    for i, item in enumerate(data):
        if labels[i] is not None:
            pattern = pm.get_pattern(labels[i])
            iou.append(np.intersect1d(pattern, item).size/(np.union1d(pattern, item).size + EPS))
        else:
            iou.append(0)
    return sum(iou)/len(iou)


def generate_data(input_size, n_patterns, sparsity):
    data = [np.random.choice(np.arange(0, input_size), max(int(input_size * sparsity), 1), replace=False) for _ in range(n_patterns)]
    return data


def main():
    input_size = 1000
    epochs = 20
    seed = 5436
    n_patterns = 1000
    sparsity = 0.05
    config = dict(
        input_size=input_size,
        max_segments=1000,
        min_distance=0.1,
        permanence_increment=0.1,
        permanence_decrement=0.01,
        segment_decrement=0.1,
        permanence_connected_threshold=0.5,
        seed=seed
    )
    data = generate_data(input_size, n_patterns, sparsity)
    pm = PatternMemory(**config)
    labels = train(pm, data, epochs, input_size, noise=0.09)
    mean_iou = test_retrieval(pm, data, labels)

    print(mean_iou)


if __name__ == '__main__':
    main()
