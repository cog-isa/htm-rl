from typing import Union

from htm_rl.common.sdr_decoders import DecoderStack, IntBucketDecoder

import numpy as np


class PulseAdapter:
    def __init__(self,
                 environment,
                 n_joints,
                 speed_delta,
                 time_delta,
                 bucket_size,
                 default_value,
                 seed):
        self.environment = environment
        self.n_joints = n_joints
        self.speed_delta = speed_delta
        self.time_delta = time_delta
        self.decoder_stack = DecoderStack()
        shift = 0
        for joint in range(self.n_joints):
            self.decoder_stack.add_decoder(
                IntBucketDecoder(3,
                                 bucket_size,
                                 default_value=default_value,
                                 seed=seed),
                bit_range=(shift, shift+3*bucket_size))
            shift += 3*bucket_size

        self.current_speeds = np.zeros(n_joints)
        self.speed_deltas = np.array([0, speed_delta, -speed_delta])

    def adapt(self, action):
        actions = self.decoder_stack.decode(action)
        current_angles = self.environment.get_current_angles()

        # speed update
        speed_deltas = self.speed_deltas[actions]
        self.current_speeds += speed_deltas

        # angle update
        next_angles = current_angles + self.current_speeds*self.time_delta

        return next_angles

    def reset(self):
        self.current_speeds = np.zeros_like(self.current_speeds)


class BioGwLabAdapter:
    def __init__(self, n_actions, bucket_size, default_value, seed):
        self.n_actions = n_actions
        self.decoder = IntBucketDecoder(n_actions, bucket_size, default_value=default_value, seed=seed)

    def adapt(self, action):
        return int(self.decoder.decode(action))

    def reset(self):
        pass
