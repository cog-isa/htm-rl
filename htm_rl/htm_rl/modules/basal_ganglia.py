from typing import List

import copy
import numpy as np
from numpy.random._generator import Generator
from htm_rl.common.sdr import SparseSdr


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


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

        self.values = None
        self.previous_stimulus = None
        self.previous_response = None
        self.current_stimulus = None
        self.current_response = None

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

    def learn(self, reward, k: int = 1, external_value: float = 0):
        """
        main Striatum learning function
        :param reward: accumulated reward since previous response (for elementary actions it's just immediate reward)
        :param k: number of steps taken after previous response (>=1 for non-elementary actions)
        :param external_value: value from lower level BG (may be !=0 for non-elementary actions)
        :return:
        """
        if (self.previous_response is not None) and (len(self.previous_response) > 0) and (
                len(self.previous_stimulus) > 0):
            value = external_value
            prev_values = np.mean(
                (self.w_d1[self.previous_response] - self.w_d2[self.previous_response])[:, self.previous_stimulus],
                axis=-1)

            if (self.current_response is not None) and (len(self.current_response) > 0) and (
                    len(self.current_stimulus) > 0):
                values = np.mean(
                    (self.w_d1[self.current_response] - self.w_d2[self.current_response])[:, self.current_stimulus],
                    axis=-1)
                value = np.median(values)

            deltas = (reward / len(self.previous_response) + (self.discount_factor ** k) * value) - prev_values

            self.w_d1[self.previous_response.reshape((-1, 1)), self.previous_stimulus] += (
                    self.alpha * deltas).reshape((-1, 1))
            self.w_d2[self.previous_response.reshape((-1, 1)), self.previous_stimulus] -= (
                    self.beta * deltas).reshape((-1, 1))

    def reset(self):
        self.previous_response = None
        self.previous_stimulus = None
        self.current_response = None
        self.current_stimulus = None
        self.values = None


class GPi:
    def __init__(self, input_size: int, output_size: int, seed: int):
        self._input_size = input_size
        self._output_size = output_size
        self._rng = np.random.default_rng(seed)
        # TODO убрать?
        assert self._input_size == self._output_size, f"Input and output have different sizes: input: {input_size}, output {output_size}"

    def compute(self, exc_input: float, inh_input) -> np.ndarray:
        out = exc_input - inh_input[0] - inh_input[1]
        out = (out - out.min()) / (out.max() - out.min() + 1e-12)
        out = self._rng.random(self._output_size) < out
        return out


class GPe:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        # TODO убрать?
        assert self._input_size == self._output_size, f"Input and output have different sizes: input: {input_size}, output {output_size}"

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
        return np.mean(self.weights / self.time)


class Thalamus:
    def __init__(self, input_size: int, output_size: int, seed: int):
        self._input_size = input_size
        self._output_size = output_size
        self._rng = np.random.default_rng(seed)
        self.response_activity = None
        if self._input_size != self._output_size:
            raise ValueError

    def compute(self, responses, responses_boost, modulation):
        activity = np.zeros(len(responses))
        bs = ~modulation
        for ind, response in enumerate(responses):
            activity[ind] = np.sum(bs[response])
        if responses_boost is not None:
            activity += responses_boost * activity.max()
        self.response_activity = activity
        probs = softmax(activity)
        out = self._rng.choice(len(activity), 1, p=probs)[0]
        return out, responses[out]


class BasalGanglia:
    alpha: float
    beta: float
    discount_factor: float
    _rng: Generator

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 alpha: float,
                 beta: float,
                 discount_factor: float,
                 seed: int):
        self._input_size = input_size
        self._output_size = output_size

        self.stri = Striatum(input_size, output_size, discount_factor, alpha, beta)
        self.stn = STN(input_size, input_size)
        self.gpi = GPi(output_size, output_size, seed)
        self.gpe = GPe(output_size, output_size)
        self.tha = Thalamus(output_size, output_size, seed)

    def reset(self):
        self.stri.reset()

    def compute(self, stimulus,
                responses: List[SparseSdr],
                responses_boost: np.ndarray = None,
                learn=True):
        d1, d2 = self.stri.compute(stimulus)
        stn = self.stn.compute(stimulus, learn=learn)
        gpe = self.gpe.compute(stn, d2)
        gpi = self.gpi.compute(stn, (d1, gpe))

        response_index, response = self.tha.compute(responses, responses_boost, gpi)
        responses_values = np.zeros(len(responses))
        for ind, resp in enumerate(responses):
            responses_values[ind] = np.median(self.stri.values[resp])

        return response_index, response, responses_values

    def force_dopamine(self, reward: float, k: int = 0, external_value: float = 0):
        self.stri.learn(reward, k, external_value)
