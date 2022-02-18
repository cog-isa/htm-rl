from typing import Union

from htm_rl.common.sdr_decoders import DecoderStack, IntBucketDecoder
from htm_rl.common.sdr_encoders import RangeDynamicEncoder, VectorDynamicEncoder
from htm_rl.envs.coppelia.environment import ArmEnv
from htm_rl.modules.v1 import V1

from math import pi

import numpy as np


class PulseActionAdapter:
    def __init__(self,
                 environment: ArmEnv,
                 mode,
                 delta,
                 time_delta,
                 bucket_size,
                 default_value,
                 seed):
        self.mode = mode
        self.environment = environment
        self.n_joints = environment.n_joints
        self.delta = delta
        self.time_delta = time_delta
        self.decoder_stack = DecoderStack()
        shift = 0
        for joint in range(self.n_joints):
            self.decoder_stack.add_decoder(
                IntBucketDecoder(3,
                                 bucket_size,
                                 default_value=default_value,
                                 seed=seed),
                bit_range=(shift, shift + 3 * bucket_size))
            shift += 3 * bucket_size

        self.current_speeds = np.zeros(self.n_joints)
        self.speed_limit = self.environment.get_joints_speed_limit()
        self.deltas = np.array([0, pi*delta / 180, -pi*delta / 180])

    def adapt(self, action):
        actions = self.decoder_stack.decode(action)
        current_angles = self.environment.get_joint_positions()
        deltas = self.deltas[actions]
        if self.mode == 'speed':
            # speed update
            self.current_speeds += deltas
            self.current_speeds = np.clip(self.current_speeds, -self.speed_limit, self.speed_limit)
            next_angles = current_angles + self.current_speeds * self.time_delta
        else:
            next_angles = current_angles + deltas

        return next_angles

    def reset(self):
        self.current_speeds = np.zeros_like(self.current_speeds)


class PulseObsAdapter:
    def __init__(self, environment: ArmEnv, config):
        self.environment = environment
        self.n_joints = self.environment.n_joints
        # setup encoders
        self.encoders = dict()
        self.observations = list()
        self.output_sdr_size = 0

        if 'camera' in self.environment.observation:
            self.encoders['camera'] = V1(self.environment.camera.get_resolution(),
                                         config['v1']['complex'],
                                         *config['v1']['simple'])
            self.observations.append('camera')
        if 'joint_pos' in self.environment.observation:
            joint_pos_range = self.environment.agent.get_joint_intervals()[1][0]
            joint_vel_limit = self.environment.get_joints_speed_limit()
            self.encoders['joint_pos'] = VectorDynamicEncoder(
                self.n_joints,
                RangeDynamicEncoder(max_value=joint_pos_range[1],
                                    min_value=joint_pos_range[0],
                                    max_speed=joint_vel_limit,
                                    min_speed=-joint_vel_limit,
                                    **config['joint_pos'])
            )
            self.observations.append('joint_pos')
        if 'joint_vel' in self.environment.observation:
            joint_vel_limit = self.environment.get_joints_speed_limit()
            self.encoders['joint_vel'] = VectorDynamicEncoder(
                self.n_joints,
                RangeDynamicEncoder(
                    max_value=joint_vel_limit,
                    min_value=-joint_vel_limit,
                    **config['joint_vel'])
            )
            self.observations.append('joint_vel')
        if 'target_pos' in self.environment.observation:
            self.encoders['target_pos'] = VectorDynamicEncoder(
                3,
                RangeDynamicEncoder(
                    cyclic=False,
                    **config['target_pos'])
            )
            self.observations.append('target_pos')
        if 'target_vel' in self.environment.observation:
            self.encoders['target_vel'] = VectorDynamicEncoder(
                3,
                RangeDynamicEncoder(
                    max_speed=1,
                    min_speed=0,
                    cyclic=False,
                    **config['target_vel'])
            )
            self.observations.append('target_vel')

        for encoder in self.encoders.values():
            self.output_sdr_size += encoder.output_sdr_size

    def adapt(self, obs):
        outputs = list()
        shift = 0
        for i, obs_type in enumerate(self.observations):
            encoder = self.encoders[obs_type]
            if obs_type == 'camera':
                sparse, _ = encoder.compute(obs[i])
                sparse = np.concatenate(sparse)
            elif obs_type == 'joint_pos':
                sparse = encoder.encode(obs[i], self.environment.get_joint_velocities())
            elif obs_type == 'joint_vel':
                sparse = encoder.encode(obs[i], np.zeros_like(obs[i]))
            elif obs_type == 'target_pos':
                sparse = encoder.encode(obs[i], self.environment.target.get_velocity())
            elif obs_type == 'target_vel':
                sparse = encoder.encode(obs[i], np.zeros_like(obs[i]))
            else:
                raise ValueError

            sparse += shift
            outputs.append(sparse)
            if encoder is not None:
                shift += encoder.output_sdr_size
        return np.concatenate(outputs)


class BioGwLabActionAdapter:
    def __init__(self, n_actions, bucket_size, default_value, seed):
        self.n_actions = n_actions
        self.decoder = IntBucketDecoder(n_actions, bucket_size, default_value=default_value, seed=seed)

    def adapt(self, action):
        return int(self.decoder.decode(action))

    def reset(self):
        pass


class BioGwLabObsAdapter:
    def adapt(self, obs):
        return obs
