import os.path
import pathlib

import numpy as np
import imageio
import random
import matplotlib.pyplot as plt
import wandb

from hima.agents.hima.hierarchy import Hierarchy, Block, InputBlock
from hima.modules.htm.pattern_memory import SpatialMemory
from hima.modules.pmc import ThaPMCToM1
from hima.modules.basal_ganglia import BasalGanglia, DualBasalGanglia, BGPMCProxy
from htm.bindings.algorithms import SpatialPooler
from hima.modules.htm.temporal_memory import ApicalBasalFeedbackTM
from hima.common.scenario import Scenario
from hima.agents.hima.hima import HIMA
from hima.agents.hima.adapters import ArmActionAdapter, ArmObsAdapter, ArmContinuousActionAdapter
from hima.envs.coppelia.environment import ArmEnv


class ArmHIMARunner:
    def __init__(self, config, logger=None, logger_config=None):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)

        block_configs = config['blocks']
        input_block_configs = config['input_blocks']

        use_intrinsic_reward = config['agent']['use_intrinsic_reward']

        blocks = list()
        print('Hierarchy ...')
        for block_conf in input_block_configs:
            blocks.append(InputBlock(**block_conf))

        for block_conf in block_configs:
            tm = ApicalBasalFeedbackTM(**block_conf['tm'])

            if block_conf['sm'] is not None:
                sm = SpatialMemory(**block_conf['sm'])
            else:
                sm = None

            if block_conf['sp'] is not None:
                sp = SpatialPooler(**block_conf['sp'])
            else:
                sp = None

            if block_conf['bg'] is not None:
                if use_intrinsic_reward:
                    bg = DualBasalGanglia(**block_conf['bg'])
                elif block_conf['bg']['continuous_action']:
                    pmc = ThaPMCToM1(**block_conf['pmc'])
                    bg = BGPMCProxy(pmc, block_conf['bg'])
                else:
                    bg = BasalGanglia(**block_conf['bg'])
            else:
                bg = None

            blocks.append(Block(tm=tm, sm=sm, sp=sp, bg=bg, **block_conf['block']))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])
        print('Agent ...')

        if 'scenario' in config.keys():
            self.scenario = Scenario(config['scenario'], self)
        else:
            self.scenario = None

        self.agent = HIMA(config['agent'], hierarchy)

        self.env_config = config['environment']
        self.environment_type = config['environment_type']
        print('Environment')
        self.workspace_limits = config['workspace_limits']
        self.environment = ArmEnv(workspace_limits=self.workspace_limits, **config['environment'])
        if 'action_adapter' in config.keys():
            self.action_adapter = ArmActionAdapter(self.environment, **config['action_adapter'])
        elif 'action_adapter_continuous' in config.keys():
            self.action_adapter = ArmContinuousActionAdapter(
                self.agent,
                self.workspace_limits,
                environment=self.environment,
                **config['action_adapter_continuous']
            )
        else:
            raise ValueError('Adapter config is not specified!')
        self.observation_adapter = ArmObsAdapter(self.environment, config['observation_adapter'])

        self.current_action = None
        self.logger = logger
        self.total_reward = 0
        self.steps = 0
        self.episode = 0
        self.animation = False
        self.running = False
        print('Visuals ...')
        self.path_to_store_logs = config['path_to_store_logs']
        pathlib.Path(self.path_to_store_logs).mkdir(parents=True, exist_ok=True)

        self.n_blocks = len(self.agent.hierarchy.blocks)
        self.block_metrics = {'anomaly_threshold': [0] * self.n_blocks,
                              'confidence_threshold': [0] * self.n_blocks,
                              'reward_modulation': [0] * self.n_blocks,
                              'da_1lvl': 0,
                              'dda_1lvl': 0,
                              'da_2lvl': 0,
                              'dda_2lvl': 0,
                              'priority_ext_1lvl': 0,
                              'priority_int_1lvl': 0,
                              'priority_ext_2lvl': 0,
                              'priority_int_2lvl': 0}
        self.logger_config = logger_config
        self.seed = seed
        self.rng = random.Random(self.seed)

    def run_episodes(self):
        print('Starting run ...')
        self.total_reward = 0
        self.steps = 0
        self.episode = 0
        self.animation = False
        self.running = True

        while True:
            if self.scenario is not None:
                self.scenario.check_conditions()

            if not self.running:
                break

            reward, obs, is_first = self.environment.observe()

            obs = self.observation_adapter.adapt(obs)

            if self.logger is not None:
                self.log(is_first)

            if is_first:
                # Ad hoc terminal state
                action_pattern = self.agent.make_action(obs)
                self.current_action = self.action_adapter.adapt(action_pattern)

                self.episode += 1
                self.steps = 0
                self.total_reward = 0

                self.agent.reset()
                self.action_adapter.reset()
            else:
                self.steps += 1
                self.total_reward += reward

            action_pattern = self.agent.make_action(obs)
            self.current_action = self.action_adapter.adapt(action_pattern)

            self.agent.reinforce(reward)

            self.environment.act(self.current_action)

        self.environment.shutdown()
        print('Run finished.')

    def draw_animation_frame(self, logger, episode, steps):
        pic = self.environment.camera.capture_rgb() * 255
        plt.imsave(os.path.join(self.path_to_store_logs,
                                f'{logger.id}_episode_{episode}_step_{steps}.png'), pic.astype('uint8'))
        plt.close()

    def update_block_metrics(self):
        for i, block in enumerate(self.agent.hierarchy.blocks):
            self.block_metrics['anomaly_threshold'][i] = self.block_metrics['anomaly_threshold'][i] + (
                    block.anomaly_threshold - self.block_metrics['anomaly_threshold'][i]) / (self.steps + 1)
            self.block_metrics['confidence_threshold'][i] = self.block_metrics['confidence_threshold'][i] + (
                    block.confidence_threshold - self.block_metrics['confidence_threshold'][i]) / (self.steps + 1)
            self.block_metrics['reward_modulation'][i] = self.block_metrics['reward_modulation'][i] + (
                    block.reward_modulation_signal - self.block_metrics['reward_modulation'][i]) / (self.steps + 1)

        self.block_metrics['da_1lvl'] = self.block_metrics['da_1lvl'] + (
                self.agent.hierarchy.output_block.da - self.block_metrics['da_1lvl']) / (self.steps + 1)
        self.block_metrics['dda_1lvl'] = self.block_metrics['dda_1lvl'] + (
                self.agent.hierarchy.output_block.dda - self.block_metrics['dda_1lvl']) / (self.steps + 1)
        if len(self.agent.hierarchy.blocks) > 4:
            self.block_metrics['da_2lvl'] = self.block_metrics['da_2lvl'] + (
                    self.agent.hierarchy.blocks[5].da - self.block_metrics['da_2lvl']) / (self.steps + 1)
            self.block_metrics['dda_2lvl'] = self.block_metrics['dda_2lvl'] + (
                    self.agent.hierarchy.blocks[5].dda - self.block_metrics['dda_2lvl']) / (self.steps + 1)
        if self.agent.use_intrinsic_reward:
            self.block_metrics['priority_ext_1lvl'] = self.block_metrics['priority_ext_1lvl'] + (
                    self.agent.hierarchy.output_block.bg.priority_ext - self.block_metrics['priority_ext_1lvl']) / (
                                                              self.steps + 1)
            self.block_metrics['priority_int_1lvl'] = self.block_metrics['priority_int_1lvl'] + (
                    self.agent.hierarchy.output_block.bg.priority_int - self.block_metrics['priority_int_1lvl']) / (
                                                              self.steps + 1)
            if len(self.agent.hierarchy.blocks) > 4:
                self.block_metrics['priority_ext_2lvl'] = self.block_metrics['priority_ext_2lvl'] + (
                        self.agent.hierarchy.blocks[5].bg.priority_ext - self.block_metrics['priority_ext_2lvl']) / (
                                                                  self.steps + 1)
                self.block_metrics['priority_int_2lvl'] = self.block_metrics['priority_int_2lvl'] + (
                        self.agent.hierarchy.blocks[5].bg.priority_int - self.block_metrics['priority_int_2lvl']) / (
                                                                  self.steps + 1)

    def reset_block_metrics(self):
        self.block_metrics = {'anomaly_threshold': [0] * self.n_blocks,
                              'confidence_threshold': [0] * self.n_blocks,
                              'reward_modulation': [0] * self.n_blocks,
                              'da_1lvl': 0,
                              'dda_1lvl': 0,
                              'da_2lvl': 0,
                              'dda_2lvl': 0,
                              'priority_ext_1lvl': 0,
                              'priority_int_1lvl': 0,
                              'priority_ext_2lvl': 0,
                              'priority_int_2lvl': 0}

    def set_feedback_boost_range(self, boost):
        self.agent.hierarchy.output_block.feedback_boost_range = boost

    def stop(self):
        self.running = False

    def log(self, is_first):
        # ///logging///
        if is_first:
            if self.animation:
                # log all saved frames for this episode
                self.animation = False
                with imageio.get_writer(os.path.join(self.path_to_store_logs,
                                                     f'{self.logger.id}_episode_{self.episode}.gif'),
                                        mode='I',
                                        fps=self.logger_config['animation_fps']) as writer:
                    for i in range(self.steps):
                        image = imageio.imread(os.path.join(self.path_to_store_logs,
                                                            f'{self.logger.id}_episode_{self.episode}_step_{i}.png'))
                        writer.append_data(image)
                self.logger.log(
                    {f'behavior_samples/animation': wandb.Video(
                        os.path.join(self.path_to_store_logs,
                                     f'{self.logger.id}_episode_{self.episode}.gif'),
                        fps=self.logger_config['animation_fps'],
                        format='gif')}, step=self.episode)

            if (self.logger is not None) and (self.episode > 0):
                self.logger.log(
                    {'main_metrics/steps': self.steps,
                     'reward': self.total_reward,
                     'episode': self.episode,
                     },
                    step=self.episode)
                if self.logger_config['log_segments']:
                    self.logger.log(
                        {
                            'connections/basal_segments': self.agent.hierarchy.output_block.tm.basal_connections.numSegments(),
                            'connections/apical_segments': self.agent.hierarchy.output_block.tm.apical_connections.numSegments(),
                            'connections/exec_segments': self.agent.hierarchy.output_block.tm.exec_feedback_connections.numSegments(),
                            'connections/inhib_segments': self.agent.hierarchy.output_block.tm.inhib_connections.numSegments()
                        },
                        step=self.episode)
                if self.logger_config['log_td_error']:
                    self.logger.log({'main_metrics/da_1lvl': self.block_metrics['da_1lvl'],
                                     'basal_ganglia/da_2lvl': self.block_metrics['da_2lvl'],
                                     'basal_ganglia/dda_1lvl': self.block_metrics['dda_1lvl'],
                                     'basal_ganglia/dda_2lvl': self.block_metrics['dda_2lvl']}, step=self.episode)
                if self.logger_config['log_priorities'] and self.agent.use_intrinsic_reward:
                    self.logger.log({'main_metrics/priority_ext_1lvl': self.block_metrics['priority_ext_1lvl'],
                                     'basal_ganglia/priority_ext_2lvl': self.block_metrics['priority_ext_2lvl'],
                                     'basal_ganglia/priority_int_1lvl': self.block_metrics['priority_int_1lvl'],
                                     'basal_ganglia/priority_int_2lvl': self.block_metrics['priority_int_2lvl']},
                                    step=self.episode)
                if self.logger_config['log_anomaly']:
                    anomaly_th = {f"blocks/anomaly_th_block{block_id}": an for block_id, an in
                                  enumerate(self.block_metrics['anomaly_threshold'])}
                    self.logger.log(anomaly_th, step=self.episode)
                if self.logger_config['log_confidence']:
                    confidence_th = {f"blocks/confidence_th_block{block_id}": an for block_id, an in
                                     enumerate(self.block_metrics['confidence_threshold'])}
                    self.logger.log(confidence_th, step=self.episode)
                if self.logger_config['log_modulation']:
                    modulation = {f"blocks/modulation_block{block_id}": x for block_id, x in
                                  enumerate(self.block_metrics['reward_modulation'])}
                    self.logger.log(modulation, step=self.episode)

                if self.logger_config['log_number_of_clusters'] and (self.agent.empowerment is not None):
                    self.logger.log(
                        {'empowerment/number_of_clusters': self.agent.empowerment.memory.number_of_clusters},
                        step=self.episode)

                self.reset_block_metrics()

            if ((((self.episode + 1) % self.logger_config['log_every_episode']) == 0) or (self.episode == 0)) and (
                    self.logger is not None):
                self.animation = True
            # \\\logging\\\

        # ///logging///
        if self.logger is not None:
            self.update_block_metrics()

        if self.animation:
            self.draw_animation_frame(self.logger, self.episode, self.steps)
        # \\\logging\\\
