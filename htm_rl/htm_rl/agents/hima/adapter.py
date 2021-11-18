import numpy as np


class BioGwLabAction:
    """
    Muscles adapter to BioGwLabEnvironment.
    """

    def __init__(self, muscles_size, n_actions, patterns, noise_tolerance=0.1, do_nothing_action='random'):
        self.muscles_size = muscles_size
        self.n_actions = n_actions
        self.noise_tolerance = noise_tolerance
        self.do_nothing_action = do_nothing_action

        self.patterns = np.array(patterns)

    def get_action(self, sparse_pattern):
        dense_pattern = np.zeros(self.muscles_size)
        dense_pattern[sparse_pattern] = 1

        pattern_sizes = self.patterns.sum(axis=1)
        overlaps = 1 - np.sum(np.abs(self.patterns - dense_pattern), axis=1) / (pattern_sizes + 1e-15)

        if np.any(overlaps >= (1 - self.noise_tolerance)):
            return np.argmax(overlaps)
        else:
            # do nothing action
            if self.do_nothing_action == 'random':
                return np.random.randint(self.n_actions)
            else:
                return self.do_nothing_action
