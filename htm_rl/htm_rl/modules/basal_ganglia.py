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

        self.error = 0
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

    def learn(self, reward, k: int = 1, external_value: float = 0, off_policy=False):
        """
        main Striatum learning function
        :param reward: accumulated reward since previous response (for elementary actions it's just immediate reward)
        :param k: number of steps taken after previous response (>=1 for non-elementary actions)
        :param external_value: value from lower level BG (may be !=0 for non-elementary actions)
        :param off_policy: if true, then max_response is used instead of current response
        :return:
        """
        if (self.previous_response is not None) and (len(self.previous_response) > 0) and (self.previous_stimulus is not None) and (
                len(self.previous_stimulus) > 0):
            value = external_value
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
        self.max_response = None
        if self._input_size != self._output_size:
            raise ValueError

    def compute(self, responses, responses_boost, modulation, softmax_beta: float = 1, epsilon_noise: float = 0):
        activity = np.zeros(len(responses))
        bs = ~modulation
        for ind, response in enumerate(responses):
            activity[ind] = np.sum(bs[response])
        if responses_boost is not None:
            activity += responses_boost * activity.max()

        self.response_activity = activity
        self.max_response = responses[np.nanargmax(activity)]

        probs = softmax(softmax_beta*activity)
        probs = epsilon_noise/probs.size + (1 - epsilon_noise)*probs
        out = self._rng.choice(len(activity), 1, p=probs)[0]
        return out, responses[out]

    def reset(self):
        self.response_activity = None
        self.max_response = None


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
                 off_policy: bool,
                 softmax_beta: float,
                 epsilon_noise: float,
                 seed: int):
        self._input_size = input_size
        self._output_size = output_size

        self.stri = Striatum(input_size, output_size, discount_factor, alpha, beta)
        self.stn = STN(input_size, input_size)
        self.gpi = GPi(output_size, output_size, seed)
        self.gpe = GPe(output_size, output_size)
        self.tha = Thalamus(output_size, output_size, seed)

        self.off_policy = off_policy
        self.softmax_beta = softmax_beta
        self.epsilon_noise = epsilon_noise

        self.td_error = self.stri.error

    def reset(self):
        self.stri.reset()
        self.tha.reset()

    def compute(self, stimulus,
                responses: List[SparseSdr],
                responses_boost: np.ndarray = None,
                learn=True):
        d1, d2 = self.stri.compute(stimulus)
        stn = self.stn.compute(stimulus, learn=learn)
        gpe = self.gpe.compute(stn, d2)
        gpi = self.gpi.compute(stn, (d1, gpe))

        response_index, response = self.tha.compute(responses, responses_boost, gpi, self.softmax_beta, self.epsilon_noise)
        self.stri.current_max_response = self.tha.max_response

        responses_values = np.zeros(len(responses))
        for ind, resp in enumerate(responses):
            responses_values[ind] = np.median(self.stri.values[resp])

        return response_index, response, responses_values

    def force_dopamine(self, reward: float, k: int = 0, external_value: float = 0, reward_int: float = 0):
        self.stri.learn(reward, k, external_value, self.off_policy)

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


class DualBasalGanglia:
    alpha: float
    beta: float
    discount_factor: float
    _rng: Generator

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 discount_factor: float = 0.997,
                 off_policy: bool = False,
                 softmax_beta: float = 1.0,
                 epsilon_noise: float = 0.0,
                 priority_ext: float = 1.0,
                 priority_int: float = 1.0,
                 td_error_threshold: float = 0.01,
                 priority_inc_factor: float = 1.2,
                 priority_dec_factor: float = 0.9,
                 seed: int = 0):
        """
        Basal Ganglia with two regions of Striatum. One for external reward and another for internal.

        :param input_size: size of input array

        :param output_size: size of output array

        :param alpha: learning rate for d1 receptors

        :param beta: learning rate for d2 receptors

        :param discount_factor: discount factor

        :param off_policy: for one step: if true, equivalent to Q-learning, else SARSA

        :param softmax_beta: inverse softmax temperature

        :param epsilon_noise: if not zero, adds noise to softmax equivalent to epsilon greedy exploration

        :param priority_ext: scaling external reward

        :param priority_int: scaling internal reward

        :param td_error_threshold: defines condition for decreasing external striatum priority
        too big value will make condition impossible
        too close to zero value will make it sensitive to noise

        :param priority_inc_factor: priority increase rate

        :param priority_dec_factor: priority decrease rate

        :param seed: seed
        """
        self._input_size = input_size
        self._output_size = output_size

        self.stri_ext = Striatum(input_size, output_size, discount_factor, alpha, beta)
        self.stri_int = Striatum(input_size, output_size, discount_factor, alpha, beta)
        self.priority_ext_init = priority_ext
        self.priority_int_init = priority_int
        self.priority_ext = 1
        self.priority_int = 0
        self.priority_inc_factor = priority_inc_factor
        self.priority_dec_factor = priority_dec_factor
        self.td_error_threshold = td_error_threshold

        self.stn = STN(input_size, input_size)
        self.gpi = GPi(output_size, output_size, seed)
        self.gpe = GPe(output_size, output_size)
        self.tha = Thalamus(output_size, output_size, seed)

        self.off_policy = off_policy
        self.softmax_beta = softmax_beta
        self.epsilon_noise = epsilon_noise

        self.current_max_response = None
        self.td_error = self.stri_ext.error

    def reset(self):
        self.stri_ext.reset()
        self.stri_int.reset()
        self.tha.reset()

    def compute(self, stimulus,
                responses: List[SparseSdr],
                responses_boost: np.ndarray = None,
                learn=True):
        """
        Choose response if stimulus given.

        :param stimulus: sparse representation of cortex activity
        :param responses: options among we choose, also represent motor cortex activations
        :param responses_boost: modify current policy via boosting response probabilities
        :param learn: it's better to set this to False for a debug mode, otherwise leave this to be True
        :return:
        """
        d1_ext, d2_ext = self.stri_ext.compute(stimulus)
        d1_int, d2_int = self.stri_int.compute(stimulus)
        d1 = (self.priority_ext_init * self.priority_ext * d1_ext +
              self.priority_int_init * self.priority_int * d1_int)
        d2 = (self.priority_ext_init * self.priority_ext * d2_ext +
              self.priority_int_init * self.priority_int * d2_int)

        stn = self.stn.compute(stimulus, learn=learn)
        gpe = self.gpe.compute(stn, d2)
        gpi = self.gpi.compute(stn, (d1, gpe))

        response_index, response = self.tha.compute(responses, responses_boost, gpi, self.softmax_beta,
                                                    self.epsilon_noise)
        self.current_max_response = self.tha.max_response

        responses_values = np.zeros(len(responses))
        for ind, resp in enumerate(responses):
            responses_values[ind] = np.median(self.priority_ext_init * self.priority_ext * self.stri_ext.values[resp] +
                                              self.priority_int_init * self.priority_int * self.stri_int.values[resp])

        return response_index, response, responses_values

    def force_dopamine(self, reward_ext: float, k: int = 0, external_value: float = 0, reward_int: float = 0.0):
        """
        Aggregates rewards.

        :param reward_ext: external reward
        :param reward_int: internal reward
        :param k: step (for n-step learning)
        :param external_value: used, when reward estimation by striatum is not available
        :return:
        """
        self.stri_ext.learn(reward_ext, k, external_value, self.off_policy)
        self.stri_int.learn(reward_int, k, external_value, self.off_policy)

        # update priorities
        td_err = self.stri_ext.error.sum()
        if td_err < -self.td_error_threshold:
            self.priority_ext = self.priority_ext * self.priority_dec_factor
        else:
            self.priority_ext = min(self.priority_ext * self.priority_inc_factor, 1.0)
        self.priority_int = 1 - self.priority_ext

    def update_response(self, response):
        """
        Forces to update striatum response history. In normal situation striatum do it automatically.

        :param response: sparse array of motor cortex activity
        :return:
        """
        self.stri_ext.update_response(response)
        self.stri_int.update_response(response)

    def update_stimulus(self, stimulus):
        """
        Forces to update striatum stimulus history. In normal situation striatum do it automatically.

        :param stimulus: sparse array of sensory cortex activity
        :return:
        """
        self.stri_ext.update_stimulus(stimulus)
        self.stri_int.update_stimulus(stimulus)
