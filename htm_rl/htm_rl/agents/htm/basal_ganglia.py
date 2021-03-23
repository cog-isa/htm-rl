from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR
import copy
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class BasalGanglia:
    alpha: float
    beta: float
    gamma: float
    discount_factor: float

    def __init__(self, input_size: int, alpha: float, beta: float, gamma: float,
                 discount_factor: float, w_STN: float, sp: SpatialPooler = None, value_window=10,
                 greedy=False, eps=None, learn_sp=True, seed=None):
        np.random.seed(seed)

        self.sp = sp
        self.input_size = input_size

        if self.sp is not None:
            self.output_size = self.sp.getColumnDimensions()
        else:
            self.output_size = input_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.discount_factor = discount_factor
        self.w_STN = w_STN

        self._D1 = 0.55 * np.ones(self.output_size)
        self._D2 = 0.45 * np.ones(self.output_size)
        self._STN = np.zeros(self.output_size)
        self._BS = None
        self._pBS = None

        self.output_values = [0]*value_window
        self.inhib_threshold = 0
        self.value_window = value_window

        self.learn_sp = learn_sp
        self.greedy = greedy
        self.eps = eps

    def reset(self):
        self._BS = None
        self._pBS = None

    def choose(self, options, condition: SDR, return_option_value=False, greedy=None, eps=None, option_weights=None, return_values=False, return_index=False):
        conditioned_options = list()
        input_sp = SDR(self.input_size)
        output_sp = SDR(self.sp.getColumnDimensions())
        for option in options:
            input_sp.sparse = np.concatenate([condition.sparse, option + condition.size])
            self.sp.compute(input_sp, self.learn_sp, output_sp)
            conditioned_options.append(np.copy(output_sp.sparse))

        active_input = np.unique(conditioned_options)

        cortex = np.zeros(self.output_size, dtype=np.int8)
        cortex[active_input] = 1

        self._STN = self._STN * (1 - self.gamma) + cortex * self.gamma
        GPi = - (self._D1 * cortex - self._D2 * cortex)
        GPi = (GPi - np.min(GPi)) / (np.max(GPi) - np.min(GPi))
        GPi = self.w_STN * np.mean(self._STN) + (1 - self.w_STN) * GPi
        GPi = np.random.random(GPi.shape) < GPi
        BS = cortex & ~GPi

        value_options = np.zeros(len(conditioned_options))
        for ind, conditioned_option in enumerate(conditioned_options):
            value_options[ind] = np.sum(BS[conditioned_option])

        if option_weights is not None:
            weighted_value_options = value_options * option_weights
        else:
            weighted_value_options = value_options

        if greedy is None:
            greedy = self.greedy

        if eps is None:
            eps = self.eps

        if greedy:
            if eps is not None:
                gamma = np.random.random()
                if gamma < eps:
                    option_index = np.random.randint(len(value_options))
                else:
                    max_value = weighted_value_options.max()
                    max_indices = np.flatnonzero(weighted_value_options == max_value)
                    option_index = np.random.choice(max_indices, 1)[0]
            else:
                max_value = weighted_value_options.max()
                max_indices = np.flatnonzero(weighted_value_options == max_value)
                option_index = np.random.choice(max_indices, 1)[0]
        else:
            option_probs = softmax(weighted_value_options)
            option_index = np.random.choice(len(conditioned_options), 1, p=option_probs)[0]

        self._BS = np.where(BS)[0]
        self._BS = np.intersect1d(self._BS, conditioned_options[option_index])
        # moving average inhibition threshold
        option_value = value_options[option_index]
        self.inhib_threshold = self.inhib_threshold + (
                option_value - self.output_values[0]) / self.value_window
        self.output_values.append(option_value)
        self.output_values.pop(0)

        value_options -= self.inhib_threshold
        value_options /= conditioned_options[0].size

        answer = [options[option_index]]
        for flag, result in zip((return_option_value, return_values, return_index), (value_options[option_index], value_options, option_index)):
            if flag:
                answer.append(result)
        return answer

    def force_dopamine(self, reward: float):
        if self._pBS is not None:
            d21 = self._D2 - self._D1
            value = 0
            if self._BS is not None:
                if len(self._BS) != 0:
                    value = -np.mean(d21[self._BS])
            delta = d21 + reward + self.discount_factor * value
            self._D1[self._pBS] = self._D1[self._pBS] + self.alpha * delta[self._pBS]
            self._D2[self._pBS] = self._D2[self._pBS] - self.beta * delta[self._pBS]
        self._pBS = copy.deepcopy(self._BS)
        self._BS = None