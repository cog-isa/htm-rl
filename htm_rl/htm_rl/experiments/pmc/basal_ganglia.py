import copy
import numpy as np
from htm_rl.common.sdr import SparseSdr

EPS = 1e-12


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1))
    return e_x / np.sum(e_x)


class Striatum:
    def __init__(self, input_size: int, output_size: int, discount_factor: float,
                 alpha: float, beta: float):
        self._input_size = input_size
        self._output_size = output_size

        self.w_d1 = np.zeros((output_size, input_size))
        self.w_d2 = np.zeros((output_size, input_size))
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.beta = beta

        self.error = np.empty(0)
        self.values = None
        self.previous_stimulus = None
        self.previous_response = None
        self.current_stimulus = None
        self.current_response = None
        self.current_max_response = None

    def compute(self, exc_input: SparseSdr) -> (np.ndarray, np.ndarray):
        if len(exc_input) > 0:
            d1 = np.mean(self.w_d1[:, exc_input], axis=-1)
            d2 = np.mean(self.w_d2[:, exc_input], axis=-1)
            self.values = d1 - d2
        else:
            self.values = np.zeros(self.w_d1.shape[0])
            d1 = np.zeros(self.w_d1.shape[0])
            d2 = np.zeros(self.w_d1.shape[0])
        return d1, d2

    def update_response(self, response: SparseSdr):
        self.previous_response = copy.deepcopy(self.current_response)
        self.current_response = copy.deepcopy(response)

    def update_stimulus(self, stimulus: SparseSdr):
        self.previous_stimulus = copy.deepcopy(self.current_stimulus)
        self.current_stimulus = copy.deepcopy(stimulus)

    def learn(self, reward, k: int = 1, off_policy=False):
        """
        main Striatum learning function
        :param reward: accumulated reward since previous response (for elementary actions it's just immediate reward)
        :param k: number of steps taken after previous response (>=1 for non-elementary actions)
        :param off_policy: if true, then max_response is used instead of current response
        :return:
        """
        if (self.previous_response is not None) and (len(self.previous_response) > 0) and (self.previous_stimulus is not None) and (
                len(self.previous_stimulus) > 0):
            value = 0
            prev_values = np.mean(
                (self.w_d1[self.previous_response] - self.w_d2[self.previous_response])[:, self.previous_stimulus],
                axis=-1)

            if (self.current_response is not None) and (len(self.current_response) > 0) and (self.current_stimulus is not None) and (
                    len(self.current_stimulus) > 0):

                if off_policy:
                    response = self.current_max_response
                else:
                    response = self.current_response

                values = np.mean(
                    (self.w_d1[response] - self.w_d2[response])[:, self.current_stimulus],
                    axis=-1)
                value = np.median(values)

            deltas = (reward / len(self.previous_response) + (self.discount_factor ** k) * value) - prev_values
            self.error = deltas

            self.w_d1[self.previous_response.reshape((-1, 1)), self.previous_stimulus] += (
                    self.alpha * deltas).reshape((-1, 1))
            self.w_d2[self.previous_response.reshape((-1, 1)), self.previous_stimulus] -= (
                    self.beta * deltas).reshape((-1, 1))

    def reset(self):
        self.previous_response = None
        self.previous_stimulus = None
        self.current_response = None
        self.current_stimulus = None
        self.current_max_response = None
        self.values = None


class GPi:
    def __init__(self, input_size: int, output_size: int, seed: int):
        self._input_size = input_size
        self._output_size = output_size
        self._rng = np.random.default_rng(seed)

    def compute(self, exc_input: float, inh_input) -> (np.ndarray, np.ndarray):
        out = exc_input - inh_input[0] - inh_input[1]
        probs = (out - out.min()) / (out.max() - out.min() + 1e-12)
        return 1-probs


class GPe:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size

    def compute(self, exc_input: float, inh_input: np.ndarray) -> np.ndarray:
        return exc_input - inh_input


class STN:
    def __init__(self, input_size: int, output_size: int):
        self._input_size = input_size
        self._output_size = output_size

        assert self._input_size == self._output_size, f"Input and output have different sizes: input: {input_size}, output {output_size}"

        self.weights = np.zeros(input_size)
        self.time = 0

    def compute(self, exc_input: SparseSdr, learn=True) -> float:
        if learn:
            self.weights[exc_input] += 1
            self.time += 1
        return float(np.mean(self.weights / self.time))


class BasalGanglia:
    alpha: float
    beta: float
    discount_factor: float

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 alpha: float,
                 beta: float,
                 discount_factor: float,
                 seed: int,
                 **kwargs):
        self._input_size = input_size
        self._output_size = output_size

        self._rng: np.random.default_rng(seed)

        self.stri = Striatum(input_size, output_size, discount_factor, alpha, beta)
        self.stn = STN(input_size, input_size)
        self.gpi = GPi(output_size, output_size, seed)
        self.gpe = GPe(output_size, output_size)

    @property
    def td_error(self):
        return self.stri.error

    def reset(self):
        self.stri.reset()

    def compute(self, stimulus,
                responses_boost: np.ndarray = None,
                learn=True):
        d1, d2 = self.stri.compute(stimulus)
        stn = self.stn.compute(stimulus, learn=learn)
        gpe = self.gpe.compute(stn, d2)
        if responses_boost is not None:
            d1 *= responses_boost
        gpi = self.gpi.compute(stn, (d1, gpe))

        return gpi

    def force_dopamine(self, reward: float, k: int = 0, **kwargs):
        self.stri.learn(reward, k)

    def update_response(self, response):
        """
        Forces to update striatum response history. In normal situation striatum do it automatically.

        :param response: sparse array of motor cortex activity
        :return:
        """
        self.stri.update_response(response)

    def update_stimulus(self, stimulus):
        """
        Forces to update striatum stimulus history. In normal situation striatum do it automatically.

        :param stimulus: sparse array of sensory cortex activity
        :return:
        """
        self.stri.update_stimulus(stimulus)
