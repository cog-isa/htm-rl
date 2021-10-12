from typing import List

import copy
import numpy as np
from numpy.random._generator import Generator
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
                 seed: int,
                 **kwargs):
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

    @property
    def td_error(self):
        return self.stri.error

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

    def force_dopamine(self, reward: float, k: int = 0, **kwargs):
        self.stri.learn(reward, k, self.off_policy)

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
                 alpha_int: float = 0.1,
                 beta_int: float = 0.1,
                 discount_factor: float = 0.997,
                 discount_factor_int: float = 0.997,
                 off_policy: bool = True,
                 off_policy_int: bool = True,
                 softmax_beta: float = 1.0,
                 epsilon_noise: float = 0.0,
                 priority_ext: float = 1.0,
                 priority_int: float = 1.0,
                 td_error_threshold: float = 0.01,
                 priority_inc_factor: float = 1.2,
                 priority_dec_factor: float = 0.9,
                 use_reward_modulation: bool = False,
                 min_reward_decay: float = 0.99,
                 max_reward_decay: float = 0.99,
                 sm_max_reward: float = 0.9,
                 sm_min_reward: float = 0.9,
                 sm_reward_inc: float = 0.9,
                 sm_reward_dec: float = 0.99,
                 intrinsic_decay: float = 0.999,
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
        self.stri_int = Striatum(input_size, output_size, discount_factor_int, alpha_int, beta_int)
        self.priority_ext_init = priority_ext
        self.priority_int_init = priority_int
        self.priority_ext = 1
        self.priority_int = 0
        self.priority_inc_factor = priority_inc_factor
        self.priority_dec_factor = priority_dec_factor
        self.td_error_threshold = td_error_threshold
        self.use_reward_modulation = use_reward_modulation

        self.stn = STN(input_size, input_size)
        self.gpi = GPi(output_size, output_size, seed)
        self.gpe = GPe(output_size, output_size)
        self.tha = Thalamus(output_size, output_size, seed)

        self.off_policy = off_policy
        self.off_policy_int = off_policy_int
        self.softmax_beta = softmax_beta
        self.epsilon_noise = epsilon_noise

        self.responses_values_ext = np.empty(0)
        self.responses_values_int = np.empty(0)

        self.mean_reward = 0
        self.max_reward = 0
        self.min_reward = 0
        self.sm_reward_inc = sm_reward_inc
        self.sm_reward_dec = sm_reward_dec
        self.max_reward_decay = max_reward_decay
        self.min_reward_decay = min_reward_decay
        self.sm_max_reward = sm_max_reward
        self.sm_min_reward = sm_min_reward
        self.intrinsic_off = 1
        self.intrinsic_decay = intrinsic_decay

        self.reward_modulation_signal = 1

    @property
    def td_error(self):
        return self.stri_ext.error

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
              self.intrinsic_off * self.priority_int_init * self.priority_int * d1_int)
        d2 = (self.priority_ext_init * self.priority_ext * d2_ext +
              self.intrinsic_off * self.priority_int_init * self.priority_int * d2_int)

        stn = self.stn.compute(stimulus, learn=learn)
        gpe = self.gpe.compute(stn, d2)
        gpi = self.gpi.compute(stn, (d1, gpe))

        response_index, response = self.tha.compute(responses, responses_boost, gpi, self.softmax_beta,
                                                    self.epsilon_noise)

        self.responses_values_ext = np.zeros(len(responses))
        self.responses_values_int = np.zeros(len(responses))
        for ind, resp in enumerate(responses):
            self.responses_values_ext[ind] = np.median(self.stri_ext.values[resp])
            self.responses_values_int[ind] = np.median(self.stri_int.values[resp])

        self.stri_ext.current_max_response = responses[np.argmax(self.responses_values_ext)]
        self.stri_int.current_max_response = responses[np.argmax(self.responses_values_int)]

        responses_values = (self.responses_values_ext * self.priority_ext_init * self.priority_ext +
                            self.responses_values_int * self.priority_int_init * self.priority_int)
        return response_index, response, responses_values

    def force_dopamine(self, reward_ext: float, k: int = 0, reward_int: float = 0.0):
        """
        Aggregates rewards.

        :param reward_ext: external reward
        :param reward_int: internal reward
        :param k: step (for n-step learning)
        :return:
        """
        self.stri_ext.learn(reward_ext, k, self.off_policy)
        self.stri_int.learn(reward_int, k, self.off_policy_int)

        # update priorities
        if self.use_reward_modulation:
            self.update_reward_modulation_signal(reward_ext)
            self.priority_ext = self.reward_modulation_signal
        else:
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

    def update_reward_modulation_signal(self, reward):
        if reward > self.mean_reward:
            self.mean_reward = self.mean_reward * self.sm_reward_inc + reward * (1 - self.sm_reward_inc)
        else:
            self.mean_reward = self.mean_reward * self.sm_reward_dec + reward * (1 - self.sm_reward_dec)

        if self.mean_reward > self.max_reward:
            self.max_reward = self.max_reward * self.sm_max_reward + self.mean_reward * (1 - self.sm_max_reward)
        else:
            self.max_reward *= self.max_reward_decay

        if self.mean_reward < self.min_reward:
            self.min_reward = self.min_reward * self.sm_min_reward + self.mean_reward * (1 - self.sm_min_reward)
        else:
            self.min_reward *= self.min_reward_decay

        if abs(self.max_reward) < EPS:
            self.reward_modulation_signal = 0
        else:
            self.reward_modulation_signal = np.clip((self.mean_reward - self.min_reward) / self.max_reward, 0.0,
                                                    1.0)

        if self.reward_modulation_signal < EPS:
            self.intrinsic_off *= self.intrinsic_decay
        else:
            self.intrinsic_off = 1
