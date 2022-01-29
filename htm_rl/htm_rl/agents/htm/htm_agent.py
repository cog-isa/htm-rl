import os.path
import pathlib
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from htm_rl.agents.htm.configurator import configure
from htm_rl.agents.htm.hierarchy import Hierarchy, Block, InputBlock, SpatialMemory
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.agents.htm.utils import (
    OptionVis, draw_values, compute_q_policy, compute_mu_policy, draw_policy,
    draw_dual_values, EmpowermentVis, get_unshifted_pos, clip_mask
)
from htm_rl.common.utils import safe_divide
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.modules.basal_ganglia import BasalGanglia, DualBasalGanglia
from htm_rl.modules.dreaming.dreamer import Dreamer
from htm_rl.modules.empowerment import Empowerment
from htm_rl.scenarios.utils import parse_str


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


class HTMAgent:
    def __init__(self, config, hierarchy: Hierarchy):
        self.use_intrinsic_reward = config['use_intrinsic_reward']
        self.use_dreaming = config['use_dreaming']
        self.punish_intrinsic_reward = config['punish_intrinsic_reward']
        self.hierarchy = hierarchy

        self.action = BioGwLabAction(**config['action'])

        self.muscles = Muscles(**config['muscles'])

        if self.use_intrinsic_reward:
            self.empowerment_horizon = config['empowerment'].pop('horizon')
            self.empowerment = Empowerment(**config['empowerment'])
        else:
            self.empowerment_horizon = 0
            self.empowerment = None

        if self.use_dreaming:
            self.dreamer = Dreamer(
                n_actions=self.action.n_actions,
                agent=self,
                state_encoder=self.hierarchy.visual_block.sp,
                **config['dreaming']
            )
        else:
            self.dreamer = None

        self.total_patterns = 2 ** self.action.muscles_size
        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        # proportionality for random generator
        self.alpha = config['alpha']

        self.sp_output = SDR(self.hierarchy.output_block.basal_columns)
        self.sp_input = SDR(self.hierarchy.output_block.get_in_sizes()[-1])
        self.previous_obs = np.empty(0)
        self.backup = None
        self.real_pos = None

    def make_action(self, state_pattern):
        self.state_pattern.sparse = state_pattern
        # get action from hierarchy
        self.hierarchy.set_input((state_pattern, self.action_pattern))
        hierarchy_action_pattern = self.hierarchy.output_block.get_output('feedback')
        # train muscles
        self.muscles.set_active_muscles(self.action_pattern)
        self.muscles.set_active_input(self.hierarchy.output_block.get_output('basal'))
        self.muscles.depolarize_muscles()
        self.muscles.learn()
        # get muscles activations
        self.muscles.set_active_input(hierarchy_action_pattern)
        self.muscles.depolarize_muscles()
        action_pattern = self.muscles.get_depolarized_muscles()

        self.action_pattern = action_pattern
        # convert muscles activation pattern to environment action
        action = self.action.get_action(action_pattern)

        # train empowerment tm
        # if self.use_intrinsic_reward:
        #     current_obs = self.hierarchy.visual_block.get_output('basal')
        #     if (self.previous_obs.size > 0) and (current_obs.size > 0) and self.empowerment.filename is None:
        #         self.empowerment.learn(self.previous_obs, current_obs)
        #     self.previous_obs = current_obs
        return action

    def get_intrinsic_reward(self):
        if self.empowerment.filename is None:
            state = self.hierarchy.visual_block.get_output('basal')
            reward = self.empowerment.eval_state(state, self.empowerment_horizon,
                                             use_memory=True)[0]
        else:

            reward = self.empowerment.eval_from_file(self.real_pos)
        return reward

    def reinforce(self, reward: float):
        """
        Reinforce BasalGanglia.
        :param reward: float:
        Main reward of the environment.
        Rewards for all blocks in hierarchy, they may differ from actual reward.
        List should be length of number of blocks in hierarchy.
        :return:
        """
        if self.use_intrinsic_reward:
            reward_int = self.get_intrinsic_reward() + self.punish_intrinsic_reward
        else:
            reward_int = 0

        self.hierarchy.add_rewards([reward] * len(self.hierarchy.blocks),
                                   [reward_int] * len(self.hierarchy.blocks))

    def reset(self):
        self.hierarchy.reset()
        self.action_pattern = np.empty(0)
        self.state_pattern.sparse = np.empty(0)
        self.previous_obs = np.empty(0)

    def generate_patterns(self):
        """
        Generate random muscles activation patterns. Number of patterns proportional to patterns in memory.
        :return: numpy array
        """
        n_patterns_to_gen = np.clip(self.alpha * (self.total_patterns - len(self.hierarchy.output_block.sm)),
                                    a_min=0, a_max=self.total_patterns)
        if n_patterns_to_gen > 0:
            return np.random.choice([0, 1], size=(int(n_patterns_to_gen), self.muscles.muscles_size // 2))
        else:
            return np.empty(0)

    def train_patterns(self, n_steps=1, train_muscles=True, train_memory=True):
        patterns = self.action.patterns
        for step in range(n_steps):
            for pattern in patterns:
                # train memory
                if self.hierarchy.output_block.sp is not None:
                    self.sp_input.dense = pattern
                    learn = self.hierarchy.output_block.learn_sp
                    self.hierarchy.output_block.sp.compute(self.sp_input, learn, self.sp_output)
                    if train_memory:
                        self.hierarchy.output_block.sm.add(self.sp_output.dense.copy())
                elif train_memory:
                    self.hierarchy.output_block.sm.add(pattern)
                    self.sp_input.dense = pattern
                    self.sp_output.dense = pattern

                # train muscles
                if train_muscles:
                    self.muscles.set_active_muscles(self.sp_input.sparse)
                    self.muscles.set_active_input(self.sp_output.sparse)
                    self.muscles.depolarize_muscles()
                    self.muscles.learn()

            self.hierarchy.output_block.sm.forget()

    def backup_agent(self):
        pass

    def restore_agent(self):
        pass


class HTMAgentRunner:
    def __init__(self, config, logger=None):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)

        block_configs = config['blocks']
        input_block_configs = config['input_blocks']

        use_intrinsic_reward = config['agent']['use_intrinsic_reward']

        blocks = list()

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
                else:
                    bg = BasalGanglia(**block_conf['bg'])
            else:
                bg = None

            blocks.append(Block(tm=tm, sm=sm, sp=sp, bg=bg, **block_conf['block']))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])

        if 'scenario' in config.keys():
            self.scenario = Scenario(config['scenario'], self)
        else:
            self.scenario = None
        self.agent = HTMAgent(config['agent'], hierarchy)
        self.env_config = config['environment']
        self.environment = BioGwLabEnvironment(**config['environment'])
        self.terminal_pos_stat = dict()
        self.last_terminal_stat = 0
        self.total_terminals = 0
        self.logger = logger
        self.total_reward = 0
        self.animation = False
        self.agent_pos = list()
        self.level = -1
        self.task = 0
        self.steps = 0
        self.steps_per_goal = 0
        self.steps_per_task = 0
        self.steps_total = 0
        self.steps_cumulative = 0
        self.all_steps = 0
        self.episode = 0
        self.option_actions = list()
        self.option_predicted_actions = list()
        self.current_option_id = None
        self.last_option_id = None
        self.current_action = None
        self.early_stop = False
        self.map_change_indicator = 0
        self.goal_reached = True
        self.task_complete = True

        self.path_to_store_logs = config['path_to_store_logs']
        pathlib.Path(self.path_to_store_logs).mkdir(parents=True, exist_ok=True)

        self.option_stat = OptionVis(self.env_config['shape_xy'], **config['vis_options'])
        self.empowerment_vis = EmpowermentVis(self.env_config['shape_xy'], self.agent.empowerment,
                                              self.environment,
                                              self.agent.empowerment_horizon,
                                              self.agent.hierarchy.visual_block.sp)
        self.option_start_pos = None
        self.option_end_pos = None
        self.last_options_usage = dict()

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
        self.seed = seed
        self.rng = random.Random(self.seed)

    def run_episodes(self, n_episodes, train_patterns=True, log_values=False, log_policy=False,
                     log_every_episode=50,
                     log_segments=False, draw_options=False, log_terminal_stat=False, draw_options_stats=False,
                     opt_threshold=0, log_option_values=False, log_option_policy=False, log_options_usage=False,
                     log_td_error=False, log_anomaly=False, log_confidence=False, log_modulation=False,
                     log_values_int=True, log_values_ext=True, log_priorities=True, log_empowerment=False,
                     log_number_of_clusters=False, animation_fps=3):
        self.total_reward = 0
        self.steps = 0
        self.steps_per_goal = 0
        self.steps_per_task = 0
        self.steps_total = 0
        self.episode = 0
        self.task = 0
        self.animation = False
        self.agent_pos = list()
        self.goal_reached = True
        self.task_complete = True

        #logging priorities
        if self.agent.use_intrinsic_reward:
            max_reward_log = 0
            mean_reward_log = 0
            min_reward_log = 0
            priority_ext_log = 0
            intrinsic_off_log = 0

        if self.logger is not None:
            self.define_logging_metrics()

        while self.episode < n_episodes:
            if self.scenario is not None:
                self.scenario.check_conditions()

            if train_patterns:
                self.agent.train_patterns()

            reward, obs, is_first = self.environment.observe()

            self.agent.real_pos = get_unshifted_pos(
                    self.environment.env.agent.position,
                    self.environment.env.renderer.shape.top_left_point
                )

            if is_first:
                self.steps_per_goal += self.steps
                self.steps_per_task += self.steps
                self.steps_total += self.steps
                if self.logger is not None and self.goal_reached:
                    self.log_goal_complete()

                # ///logging///
                self.all_steps += self.steps
                if self.animation:
                    # log all saved frames for this episode
                    self.animation = False
                    with imageio.get_writer(os.path.join(self.path_to_store_logs,
                                                         f'{self.logger.id}_episode_{self.episode}.gif'),
                                            mode='I',
                                            fps=animation_fps) as writer:
                        for i in range(self.steps):
                            image = imageio.imread(os.path.join(self.path_to_store_logs,
                                                                f'{self.logger.id}_episode_{self.episode}_step_{i}.png'))
                            writer.append_data(image)
                    self.logger.log(
                        {f'behavior_samples/animation': wandb.Video(
                            os.path.join(self.path_to_store_logs,
                                         f'{self.logger.id}_episode_{self.episode}.gif'),
                            fps=animation_fps,
                            format='gif')}, step=self.episode)

                if (self.logger is not None) and (self.episode > 0):
                    self.logger.log(
                        {'main_metrics/steps': self.steps, 'reward': self.total_reward, 'episode': self.episode,
                         'main_metrics/level': self.level,
                         'main_metrics/total_terminals': self.total_terminals,
                         'main_metrics/steps_cumulative': self.steps_cumulative,
                         'main_metrics/total_steps': self.steps_total,
                         'main_metrics/map_change_indicator': self.map_change_indicator,
                         'main_metrics/all_steps': self.all_steps,
                         },
                        step=self.episode)
                    if self.agent.use_intrinsic_reward:
                        self.logger.log(
                            {'priority/mean_reward': mean_reward_log/self.steps,
                             'priority/max_reward': max_reward_log/self.steps,
                             'priority/min_reward': min_reward_log/self.steps,
                             'priority/priority_ext': priority_ext_log/self.steps,
                             'priority/intrinsic_off': intrinsic_off_log/self.steps},
                            step=self.episode
                        )
                    self.map_change_indicator = 0
                    if log_segments:
                        self.logger.log(
                            {
                                'connections/basal_segments': self.agent.hierarchy.output_block.tm.basal_connections.numSegments(),
                                'connections/apical_segments': self.agent.hierarchy.output_block.tm.apical_connections.numSegments(),
                                'connections/exec_segments': self.agent.hierarchy.output_block.tm.exec_feedback_connections.numSegments(),
                                'connections/inhib_segments': self.agent.hierarchy.output_block.tm.inhib_connections.numSegments()
                            },
                            step=self.episode)

                    if log_options_usage:
                        options_usage_gain = self.get_options_usage_gain()
                        self.logger.log(
                            {f"options/option_{key}_usage": value for key, value in options_usage_gain.items()},
                            step=self.episode)
                        self.logger.log({'main_metrics/total_options_usage': sum(options_usage_gain.values())},
                                        step=self.episode)
                        self.update_options_usage()

                    if log_td_error:
                        self.logger.log({'main_metrics/da_1lvl': self.block_metrics['da_1lvl'],
                                         'basal_ganglia/da_2lvl': self.block_metrics['da_2lvl'],
                                         'basal_ganglia/dda_1lvl': self.block_metrics['dda_1lvl'],
                                         'basal_ganglia/dda_2lvl': self.block_metrics['dda_2lvl']}, step=self.episode)
                    if log_priorities and self.agent.use_intrinsic_reward:
                        self.logger.log({'main_metrics/priority_ext_1lvl': self.block_metrics['priority_ext_1lvl'],
                                         'basal_ganglia/priority_ext_2lvl': self.block_metrics['priority_ext_2lvl'],
                                         'basal_ganglia/priority_int_1lvl': self.block_metrics['priority_int_1lvl'],
                                         'basal_ganglia/priority_int_2lvl': self.block_metrics['priority_int_2lvl']},
                                        step=self.episode)
                    if log_anomaly:
                        anomaly_th = {f"blocks/anomaly_th_block{block_id}": an for block_id, an in
                                      enumerate(self.block_metrics['anomaly_threshold'])}
                        self.logger.log(anomaly_th, step=self.episode)
                    if log_confidence:
                        confidence_th = {f"blocks/confidence_th_block{block_id}": an for block_id, an in
                                         enumerate(self.block_metrics['confidence_threshold'])}
                        self.logger.log(confidence_th, step=self.episode)
                    if log_modulation:
                        modulation = {f"blocks/modulation_block{block_id}": x for block_id, x in
                                      enumerate(self.block_metrics['reward_modulation'])}
                        self.logger.log(modulation, step=self.episode)

                    if log_number_of_clusters and (self.agent.empowerment is not None):
                        self.logger.log(
                            {'empowerment/number_of_clusters': self.agent.empowerment.memory.number_of_clusters},
                            step=self.episode)

                    self.reset_block_metrics()

                if ((self.episode % log_every_episode) == 0) and (self.logger is not None) and (self.episode > 0):
                    if draw_options_stats:
                        self.option_stat.draw_options(self.logger, self.episode, self.path_to_store_logs, threshold=opt_threshold,
                                                      obstacle_mask=clip_mask(
                                                          self.environment.env.entities['obstacle'].mask,
                                                          self.environment.env.renderer.shape.top_left_point,
                                                          self.env_config['shape_xy']))
                        self.option_stat.clear_stats(opt_threshold)
                        self.last_options_usage = dict()
                    if log_terminal_stat:
                        self.logger.log(
                            dict([(f'terminal_stats/{x[0]}', x[1]) for x in self.terminal_pos_stat.items()]),
                            step=self.episode)
                    if log_empowerment and (self.agent.empowerment is not None):
                        self.empowerment_vis.draw(os.path.join(self.path_to_store_logs,
                                                               f'empowerment_real_{self.logger.id}.png'),
                                                  os.path.join(self.path_to_store_logs,
                                                               f'empowerment_learned_{self.logger.id}.png'))
                        self.logger.log({
                            'empowerment/real': wandb.Image(os.path.join(self.path_to_store_logs,
                                                                               f'empowerment_real_{self.logger.id}.png')),
                            'empowerment/learned': wandb.Image(
                                os.path.join(self.path_to_store_logs,
                                             f'empowerment_learned_{self.logger.id}.png'))
                        }, step=self.episode)
                    if (log_values_ext or log_values_int) and self.agent.use_intrinsic_reward:
                        draw_dual_values(self.env_config['shape_xy'], self.environment.env, self.agent,
                                         os.path.join(self.path_to_store_logs,
                                                      f'values_ext_{self.logger.id}_{self.episode}.png'),
                                         os.path.join(self.path_to_store_logs,
                                                      f'values_int_{self.logger.id}_{self.episode}.png'))
                        if log_values_ext:
                            self.logger.log(
                                {'values/state_values_ext': wandb.Image(
                                    os.path.join(self.path_to_store_logs,
                                                 f'values_ext_{self.logger.id}_{self.episode}.png'))},
                                step=self.episode)
                        if log_values_int:
                            self.logger.log(
                                {'values/state_values_int': wandb.Image(
                                    os.path.join(self.path_to_store_logs,
                                                 f'values_int_{self.logger.id}_{self.episode}.png'))},
                                step=self.episode)

                    if (log_values or log_policy) and (not self.agent.use_intrinsic_reward):
                        if len(self.option_stat.action_displace) == 3:
                            directions = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
                            actions_map = {0: 'move', 1: 'turn_right', 2: 'turn_left'}
                        else:
                            directions = None
                            actions_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

                        q, policy, actions = compute_q_policy(self.environment.env, self.agent, directions)

                        if log_values:
                            draw_values(os.path.join(self.path_to_store_logs,
                                                     f'values_{self.logger.id}_{self.episode}.png'),
                                        self.env_config['shape_xy'],
                                        q,
                                        policy,
                                        directions=directions)
                            self.logger.log({'values/state_values': wandb.Image(
                                os.path.join(self.path_to_store_logs,
                                             f'values_{self.logger.id}_{self.episode}.png'))},
                                            step=self.episode)
                        if log_policy:
                            draw_policy(os.path.join(self.path_to_store_logs,
                                                     f'policy_{self.logger.id}_{self.episode}.png'),
                                        self.env_config['shape_xy'],
                                        policy,
                                        actions,
                                        directions=directions,
                                        actions_map=actions_map)
                            self.logger.log(
                                {'values/policy': wandb.Image(os.path.join(self.path_to_store_logs,
                                                                           f'policy_{self.logger.id}_{self.episode}.png'))},
                                step=self.episode)

                    if log_option_values or log_option_policy:
                        if len(self.option_stat.action_displace) == 3:
                            directions = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
                        else:
                            directions = None

                        q, policy, option_ids = compute_mu_policy(self.environment.env, self.agent, directions)

                        if log_option_values:
                            draw_values(os.path.join(self.path_to_store_logs,
                                                     f'option_values_{self.logger.id}_{self.episode}.png'),
                                        self.env_config['shape_xy'],
                                        q,
                                        policy,
                                        directions=directions)
                            self.logger.log({'values/option_state_values': wandb.Image(
                                os.path.join(self.path_to_store_logs,
                                             f'option_values_{self.logger.id}_{self.episode}.png'))},
                                            step=self.episode)
                        if log_option_policy:
                            draw_policy(os.path.join(self.path_to_store_logs,
                                                     f'option_policy_{self.logger.id}_{self.episode}.png'),
                                        self.env_config['shape_xy'],
                                        policy,
                                        option_ids,
                                        directions=directions)
                            self.logger.log({'values/option_policy': wandb.Image(
                                os.path.join(self.path_to_store_logs,
                                             f'option_policy_{self.logger.id}_{self.episode}.png'))},
                                            step=self.episode)

                if ((((self.episode + 1) % log_every_episode) == 0) or (self.episode == 0)) and (
                        self.logger is not None):
                    self.animation = True
                    self.agent_pos.clear()
                # \\\logging\\\

                # Ad hoc terminal state
                self.current_action = self.agent.make_action(obs)
                if self.logger is not None and self.task_complete:
                    self.log_task_complete()
                if self.early_stop:
                    break

                self.steps_cumulative += self.steps
                if self.agent.use_intrinsic_reward:
                    max_reward_log = 0
                    mean_reward_log = 0
                    min_reward_log = 0
                    priority_ext_log = 0
                    intrinsic_off_log = 0

                self.episode += 1
                self.steps = 0
                self.total_reward = 0

                if self.goal_reached:
                    self.on_new_goal()
                if self.task_complete:
                    self.on_new_task()

                self.agent.reset()
                if self.agent.use_dreaming:
                    self.agent.dreamer.on_new_episode()
            else:
                self.steps += 1
                self.total_reward += reward
                # logging
                if self.agent.use_intrinsic_reward:
                    mean_reward_log += self.agent.hierarchy.output_block.bg.mean_reward
                    max_reward_log += self.agent.hierarchy.output_block.bg.max_reward
                    min_reward_log += self.agent.hierarchy.output_block.bg.min_reward
                    priority_ext_log += self.agent.hierarchy.output_block.bg.priority_ext
                    intrinsic_off_log += self.agent.hierarchy.output_block.bg.intrinsic_off
                # logging

            if self.agent.use_dreaming and self.agent.dreamer.can_dream(reward) and self.agent.dreamer.decide_to_dream(obs):
                self.agent.dreamer.dream(obs)

            self.current_action = self.agent.make_action(obs)

            self.agent.reinforce(reward)
            if self.agent.use_dreaming:
                self.agent.dreamer.on_wake_step(obs, reward, self.current_action)

            # ///logging///
            if self.logger is not None:
                self.update_block_metrics()

            if draw_options_stats:
                self.update_option_stats(self.environment.callmethod('is_terminal'))

            if self.animation:
                self.draw_animation_frame(self.logger, draw_options, self.agent_pos, self.episode, self.steps)
            # \\\logging\\\

            self.environment.act(self.current_action)

            # ///logging///
            if self.environment.callmethod('is_terminal') and (self.environment.env.items_collected > 0):
                self.goal_reached = True
                pos = get_unshifted_pos(
                    self.environment.env.agent.position,
                    self.environment.env.renderer.shape.top_left_point
                )
                if pos in self.terminal_pos_stat:
                    self.terminal_pos_stat[pos] += 1
                else:
                    self.terminal_pos_stat[pos] = 1
                self.last_terminal_stat = self.terminal_pos_stat[pos]
            # \\\logging\\\

        if self.logger is not None:
            self.logger.log({"total_steps": self.steps_total})

    def draw_animation_frame(self, logger, draw_options, agent_pos, episode, steps):
        pic = self.environment.callmethod('render_rgb')
        if isinstance(pic, list):
            pic = pic[0]

        if draw_options:
            option_block = self.agent.hierarchy.blocks[5]
            c_pos = get_unshifted_pos(self.environment.env.agent.position,
                                      self.environment.env.renderer.shape.top_left_point)
            c_direction = self.environment.env.agent.view_direction
            c_option_id = option_block.current_option

            if self.agent.hierarchy.blocks[5].made_decision:
                if c_option_id != self.last_option_id:
                    if len(agent_pos) > 0:
                        agent_pos.clear()
                agent_pos.append(c_pos)
                if len(agent_pos) > 1:
                    pic[tuple(zip(*agent_pos))] = [[255, 255, 150]] * len(agent_pos)
                else:
                    pic[agent_pos[0]] = [255, 255, 255]
            else:
                if len(agent_pos) > 0:
                    agent_pos.clear()

            self.last_option_id = c_option_id

            term_draw_options = np.zeros((pic.shape[0], 3, 3))
            c_option = self.agent.hierarchy.blocks[5].current_option
            f_option = self.agent.hierarchy.blocks[5].failed_option
            comp_option = self.agent.hierarchy.blocks[5].completed_option
            self.agent.hierarchy.blocks[5].failed_option = None
            self.agent.hierarchy.blocks[5].completed_option = None

            if c_option is not None:
                term_draw_options[c_option, 0] = [255, 255, 255]
            if f_option is not None:
                term_draw_options[f_option, 1] = [200, 0, 0]
            if comp_option is not None:
                term_draw_options[comp_option, 2] = [0, 0, 200]

            if self.agent.hierarchy.output_block.predicted_options is not None:
                predicted_options = self.agent.hierarchy.output_block.sm.get_options_by_id(
                    self.agent.hierarchy.output_block.predicted_options)
                for o in predicted_options:
                    predicted_action_pattern = np.flatnonzero(o)
                    self.agent.muscles.set_active_input(predicted_action_pattern)
                    self.agent.muscles.depolarize_muscles()
                    action_pattern = self.agent.muscles.get_depolarized_muscles()
                    # convert muscles activation pattern to environment action
                    p_action = self.agent.action.get_action(action_pattern)
                    direction = c_direction - self.option_stat.action_rotation[p_action]
                    if direction < 0:
                        direction = 4 - direction
                    else:
                        direction %= 4
                    if (len(self.option_stat.action_displace) == 4) or (
                            np.all(self.option_stat.action_displace[p_action] == 0)):
                        displacement = self.option_stat.action_displace[p_action]
                    else:
                        displacement = self.option_stat.transform_displacement((0, 1), direction)
                    p_pos = (c_pos[0] + displacement[0], c_pos[1] + displacement[1])

                    if (p_pos[0] < pic.shape[0]) and (p_pos[1] < pic.shape[1]):
                        pic[p_pos[0], p_pos[1]] = [255, 200, 120]
            pic = np.concatenate([pic, term_draw_options], axis=1)

        plt.imsave(os.path.join(self.path_to_store_logs,
                                f'{logger.id}_episode_{episode}_step_{steps}.png'), pic.astype('uint8'))
        plt.close()

    def update_option_stats(self, is_terminal):
        option_block = self.agent.hierarchy.blocks[5]
        top_left_point = self.environment.env.renderer.shape.top_left_point

        if option_block.made_decision and not is_terminal:
            current_option_id = option_block.current_option
            if self.current_option_id != current_option_id:
                if len(self.option_actions) != 0:
                    # update stats
                    self.option_end_pos = get_unshifted_pos(self.environment.env.agent.position,
                                                            top_left_point)
                    self.option_stat.update(self.current_option_id,
                                            self.option_start_pos,
                                            self.option_end_pos,
                                            self.option_actions,
                                            self.option_predicted_actions)
                    self.option_actions.clear()
                    self.option_predicted_actions = list()

                self.option_start_pos = get_unshifted_pos(self.environment.env.agent.position,
                                                          top_left_point)

            predicted_actions = list()
            if self.agent.hierarchy.output_block.predicted_options is not None:
                predicted_options = self.agent.hierarchy.output_block.sm.get_options_by_id(
                    self.agent.hierarchy.output_block.predicted_options)
                for o in predicted_options:
                    predicted_action_pattern = np.flatnonzero(o)
                    self.agent.muscles.set_active_input(predicted_action_pattern)
                    self.agent.muscles.depolarize_muscles()
                    action_pattern = self.agent.muscles.get_depolarized_muscles()
                    # convert muscles activation pattern to environment action
                    a = self.agent.action.get_action(action_pattern)
                    predicted_actions.append(a)

                self.option_actions.append(self.current_action)
                self.option_predicted_actions.append(predicted_actions)
                self.current_option_id = current_option_id
        else:
            if len(self.option_actions) > 0:
                if option_block.current_option is not None:
                    last_option = option_block.current_option
                elif option_block.failed_option is not None:
                    last_option = option_block.failed_option
                elif option_block.completed_option is not None:
                    last_option = option_block.completed_option
                else:
                    last_option = None
                if last_option is not None:
                    last_option_id = last_option
                    self.option_end_pos = get_unshifted_pos(self.environment.env.agent.position,
                                                            top_left_point)
                    self.option_stat.update(last_option_id,
                                            self.option_start_pos,
                                            self.option_end_pos,
                                            self.option_actions,
                                            self.option_predicted_actions)
                self.option_actions.clear()
                self.option_predicted_actions = list()
                self.current_option_id = None

    def get_options_usage_gain(self):
        options_usage_gain = dict()
        for id_, stats in self.option_stat.options.items():
            if id_ in self.last_options_usage.keys():
                last_value = self.last_options_usage[id_]
            else:
                last_value = 0
            options_usage_gain[id_] = stats['n_uses'] - last_value
        return options_usage_gain

    def update_options_usage(self):
        last_options_usage = dict()
        for id_, stats in self.option_stat.options.items():
            last_options_usage[id_] = stats['n_uses']
        self.last_options_usage = last_options_usage

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

    def set_food_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.env.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        self.environment.env.modules['food'].generator.positions = positions

    def set_feedback_boost_range(self, boost):
        self.agent.hierarchy.output_block.feedback_boost_range = boost

    def set_agent_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.env.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        self.environment.env.modules['agent'].positions = positions

    def set_pos_rand_rooms(self, agent_fixed_positions=None, food_fixed_positions=None, door_positions=None, wall_thickness=1):
        """
        Room numbers:
        |1|2|
        |3|4|
        :param agent_fixed_positions:
        :param food_fixed_positions:
        :param door_positions
        :param wall_thickness:
        :return:
        """

        def ranges(room, width):
            if room < 3:
                row_range = [0, width - 1]
                if room == 1:
                    col_range = [0, width - 1]
                else:
                    col_range = [width + wall_thickness, width * 2 + wall_thickness - 1]
            else:
                row_range = [width + wall_thickness, 2 * width + wall_thickness - 1]
                if room == 3:
                    col_range = [0, width - 1]
                else:
                    col_range = [width + wall_thickness, width * 2 + wall_thickness - 1]
            return row_range, col_range

        def get_adjacent_rooms(room):
            if (room == 2) or (room == 3):
                return [1, 4]
            else:
                return [2, 3]

        adjacent_doors = {1: [1, 2], 2: [2, 3], 3: [1, 4], 4: [3, 4]}

        if self.level < 2:
            agent_room = self.rng.randint(1, 4)
            if self.level < 1:
                food_room = None
                food_door = self.rng.sample(adjacent_doors[agent_room], k=1)[0]
            else:
                food_room = self.rng.sample(get_adjacent_rooms(agent_room), k=1)[0]
                food_door = None
        else:
            agent_room, food_room = self.rng.sample(list(range(1, 5)), k=2)
            food_door = None

        room_width = (self.env_config['shape_xy'][0] - wall_thickness) // 2
        if agent_fixed_positions is not None:
            agent_pos = tuple(agent_fixed_positions[agent_room - 1])
        else:
            row_range, col_range = ranges(agent_room, room_width)
            row = self.rng.randint(*row_range)
            col = self.rng.randint(*col_range)
            agent_pos = (row, col)

        if food_door is not None:
            food_pos = tuple(door_positions[food_door - 1])
        elif food_fixed_positions is not None:
            food_pos = tuple(food_fixed_positions[food_room - 1])
        else:
            row_range, col_range = ranges(food_room, room_width)
            row = self.rng.randint(*row_range)
            col = self.rng.randint(*col_range)
            food_pos = (row, col)

        self.set_agent_positions([agent_pos])
        self.set_food_positions([food_pos])
        self.environment.callmethod('reset')
        if self.logger is not None:
            self.draw_map(self.logger)

        self.task_complete = True

    def set_pos_in_order(self, agent_positions, food_positions):
        if self.task < len(agent_positions):
            agent_pos = tuple(agent_positions[self.task])
            food_pos = tuple(food_positions[self.task])

            self.set_agent_positions([agent_pos])
            self.set_food_positions([food_pos])
        self.environment.callmethod('reset')
        if self.logger is not None:
            if self.task < len(agent_positions):
                self.draw_map(self.logger)
        self.task_complete = True

    def level_up(self):
        self.level += 1

    def stop(self):
        self.early_stop = True

    def draw_map(self, logger):
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        plt.imsave(os.path.join(self.path_to_store_logs,
                                f'map_{config["environment"]["seed"]}_{self.episode}.png'), map_image.astype('uint8'))
        plt.close()
        logger.log({'maps/map': wandb.Image(os.path.join(self.path_to_store_logs,
                                                          f'map_{config["environment"]["seed"]}_{self.episode}.png'))},
                   step=self.episode)

    def log_dreaming_stats(self):
        dreaming_stats = self.agent.dreamer.stats
        cl_wake_stats = dreaming_stats.wake_cluster_memory_stats
        cl_dreaming_stats = dreaming_stats.dreaming_cluster_memory_stats
        stats_to_log = {
            'rollouts': dreaming_stats.rollouts,
            'avg_dreaming_rate': safe_divide(dreaming_stats.times, self.steps_per_goal),
            'avg_depth': dreaming_stats.avg_depth,
            'cl_wake_match_rate': cl_wake_stats.avg_match_rate,
            'cl_wake_avg_match_similarity': cl_wake_stats.avg_match_similarity,
            'cl_wake_avg_mismatch_similarity': cl_wake_stats.avg_mismatch_similarity,
            'cl_wake_all': cl_wake_stats.matched + cl_wake_stats.mismatched,
            'cl_wake_added': cl_wake_stats.added,
            'cl_wake_removed': cl_wake_stats.removed,
            'cl_wake_avg_removed_cluster_intra_similarity': cl_wake_stats.avg_removed_cluster_intra_similarity,
            'cl_wake_avg_removed_cluster_trace': cl_wake_stats.avg_removed_cluster_trace,
            'cl_dreaming_all': cl_dreaming_stats.matched + cl_dreaming_stats.mismatched,
            'cl_dreaming_match_rate': cl_dreaming_stats.avg_match_rate,
            'cl_dreaming_avg_match_similarity': cl_dreaming_stats.avg_match_similarity,
            'cl_dreaming_avg_mismatch_similarity': cl_dreaming_stats.avg_mismatch_similarity,
        }
        for k in stats_to_log.keys():
            self.logger.log({f'dreaming/{k}': stats_to_log[k]}, step=self.episode)

        self.agent.dreamer.on_new_goal()

    def log_goal_complete(self):
        self.logger.log({
            'goal': self.total_terminals,
            'main_metrics/g_goal_steps': self.steps_per_goal,
            'main_metrics/g_task_steps': self.steps_per_task,
            'main_metrics/g_total_steps': self.steps_total,
            'main_metrics/g_episode': self.episode,
        }, step=self.episode)

        if self.agent.use_dreaming:
            self.log_dreaming_stats()

    def on_new_goal(self):
        self.goal_reached = False
        self.total_terminals += 1
        self.steps_per_goal = 0

    def log_task_complete(self):
        self.logger.log({
            'task': self.task,
            'main_metrics/steps_per_task': self.steps_per_task,
            'main_metrics/t_task_steps': self.steps_per_task,
            'main_metrics/t_total_steps': self.steps_total
        }, step=self.episode)

    def on_new_task(self):
        self.task_complete = False
        self.steps_per_task = 0
        self.task += 1
        self.map_change_indicator = 1

    @staticmethod
    def define_logging_metrics():
        wandb.define_metric("task")
        wandb.define_metric("main_metrics/steps_per_task", step_metric="task")
        wandb.define_metric("main_metrics/t_*", step_metric="task")

        wandb.define_metric("goal")
        wandb.define_metric("main_metrics/g_*", step_metric="goal")
        wandb.define_metric("dreaming/*", step_metric="goal")


class Scenario:
    def __init__(self, path, runner: HTMAgentRunner):
        self.runner = runner
        self.events = list()
        with open(path, 'r') as file:
            events = yaml.load(file, Loader=yaml.Loader)
            for event in events:
                condition = event['condition']
                check_step = event['check_every']
                action = event['action']
                params = event['params']
                self.events.append(
                    {'condition': condition, 'check_step': check_step, 'action': action, 'params': params,
                     'done': False, 'last_check': None})

    def check_conditions(self):
        for event in self.events:
            step = self.get_attr(event['check_step'])
            if (event['last_check'] != step) and not event['done']:
                event['last_check'] = step
                attr_name, operator, val, repeat = event['condition']
                attr = self.get_attr(attr_name)
                if operator == 'equal':
                    if attr == val:
                        self.execute(event)
                elif operator == 'mod':
                    if (attr % val) == 0:
                        self.execute(event)
                else:
                    raise NotImplemented(f'Operator "{operator}" is not implemented!')

    def execute(self, event):
        method_name = event['action']
        params = event['params']
        f = self.get_attr(method_name)
        f(**params)
        if event['condition'][-1] == 'norepeat':
            event['done'] = True

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        default_config_name = sys.argv[1]
    else:
        default_config_name = 'four_rooms_9x9_swap_all_together'
    with open(f'../../experiments/htm_agent/configs/{default_config_name}.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb
    else:
        logger = None

    for arg in sys.argv[2:]:
        key, value = arg.split('=')

        # accepts everything non-parseable as is, i.e as a str
        value = parse_str(value)

        key = key.lstrip('-')
        if key.endswith('.'):
            # a trick that allow distinguishing sweep params from config params
            # by adding a suffix `.` to sweep param - now we should ignore it
            key = key[:-1]
        tokens = key.split('.')
        c = config
        for k in tokens[:-1]:
            if not k:
                # a trick that allow distinguishing sweep params from config params
                # by inserting additional dots `.` to sweep param - we just ignore it
                continue
            if 0 in c:
                k = int(k)
            c = c[k]
        c[tokens[-1]] = value

    if logger is not None:
        logger = logger.init(project=config['project'], entity=config['entity'], config=config)

    # with open('../../experiments/htm_agent/htm_config_unpacked.yaml', 'w') as file:
    #     yaml.dump(configure(config), file, Dumper=yaml.Dumper)

    agent_name = config['agent']
    if agent_name in ['dqn', 'o-c']:
        from htm_rl.agents.dqn.runner import Runner
        runner = Runner(config, logger=logger)
    elif agent_name == 'hima':
        runner = HTMAgentRunner(configure(config), logger=logger)
        runner.agent.train_patterns()
    else:
        raise AttributeError('Unknown agent name in config.agent')

    # if logger is not None:
    #     runner.draw_map(logger)

    runner.run_episodes(**config['run_options'])
