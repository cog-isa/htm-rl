import os.path
import pathlib

import numpy as np
import imageio
import random
import matplotlib.pyplot as plt
import wandb

from htm_rl.agents.htm.htm_agent import Scenario
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.agents.htm.utils import OptionVis, get_unshifted_pos


class Runner:
    def __init__(self, config, logger=None):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)

        if 'scenario' in config.keys():
            self.scenario = Scenario(config['scenario'], self)
        else:
            self.scenario = None

        self.env_config = config['environment']
        self.environment = BioGwLabEnvironment(**self.env_config)
        self.agent = resolve_agent(
            config['agent'],
            state_dim=self.environment.output_sdr_size,
            action_dim=self.environment.n_actions,
            **config['cagent']
        )

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
        self.option_start_pos = None
        self.option_end_pos = None
        self.last_options_usage = dict()
        self.seed = seed
        self.rng = random.Random(self.seed)

    def run_episodes(
            self, n_episodes, train_patterns=True, log_values=False, log_policy=False,
            log_every_episode=50, draw_options=False, log_terminal_stat=False, draw_options_stats=False,
            opt_threshold=0, log_option_values=False, log_option_policy=False, log_options_usage=False,
            animation_fps=3, **_
    ):
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

        if self.logger is not None:
            self.define_logging_metrics()

        while self.episode < n_episodes:
            # print(self.steps)

            if self.scenario is not None:
                self.scenario.check_conditions()

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
                    self.map_change_indicator = 0

                    # if log_options_usage:
                    #     options_usage_gain = self.get_options_usage_gain()
                    #     self.logger.log(
                    #         {f"options/option_{key}_usage": value for key, value in options_usage_gain.items()},
                    #         step=self.episode)
                    #     self.logger.log({'main_metrics/total_options_usage': sum(options_usage_gain.values())},
                    #                     step=self.episode)
                    #     self.update_options_usage()

                if ((self.episode % log_every_episode) == 0) and (self.logger is not None) and (self.episode > 0):
                    # if draw_options_stats:
                    #     self.option_stat.draw_options(self.logger, self.episode, self.path_to_store_logs, threshold=opt_threshold,
                    #                                   obstacle_mask=clip_mask(
                    #                                       self.environment.env.entities['obstacle'].mask,
                    #                                       self.environment.env.renderer.shape.top_left_point,
                    #                                       self.env_config['shape_xy']))
                    #     self.option_stat.clear_stats(opt_threshold)
                    #     self.last_options_usage = dict()
                    if log_terminal_stat:
                        self.logger.log(
                            dict([(f'terminal_stats/{x[0]}', x[1]) for x in self.terminal_pos_stat.items()]),
                            step=self.episode)

                    # if log_values or log_policy:
                    #     if len(self.option_stat.action_displace) == 3:
                    #         directions = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
                    #         actions_map = {0: 'move', 1: 'turn_right', 2: 'turn_left'}
                    #     else:
                    #         directions = None
                    #         actions_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
                    #
                    #     q, policy, actions = compute_q_policy(self.environment.env, self.agent, directions)

                        # if log_policy:
                        #     draw_policy(os.path.join(self.path_to_store_logs,
                        #                              f'policy_{self.logger.id}_{self.episode}.png'),
                        #                 self.env_config['shape_xy'],
                        #                 policy,
                        #                 actions,
                        #                 directions=directions,
                        #                 actions_map=actions_map)
                        #     self.logger.log(
                        #         {'values/policy': wandb.Image(os.path.join(self.path_to_store_logs,
                        #                                                    f'policy_{self.logger.id}_{self.episode}.png'))},
                        #         step=self.episode)

                    # if log_option_values or log_option_policy:
                    #     if len(self.option_stat.action_displace) == 3:
                    #         directions = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
                    #     else:
                    #         directions = None
                    #
                    #     q, policy, option_ids = compute_mu_policy(self.environment.env, self.agent, directions)
                    #
                    #     if log_option_values:
                    #         draw_values(os.path.join(self.path_to_store_logs,
                    #                                  f'option_values_{self.logger.id}_{self.episode}.png'),
                    #                     self.env_config['shape_xy'],
                    #                     q,
                    #                     policy,
                    #                     directions=directions)
                    #         self.logger.log({'values/option_state_values': wandb.Image(
                    #             os.path.join(self.path_to_store_logs,
                    #                          f'option_values_{self.logger.id}_{self.episode}.png'))},
                    #                         step=self.episode)
                    #     if log_option_policy:
                    #         draw_policy(os.path.join(self.path_to_store_logs,
                    #                                  f'option_policy_{self.logger.id}_{self.episode}.png'),
                    #                     self.env_config['shape_xy'],
                    #                     policy,
                    #                     option_ids,
                    #                     directions=directions)
                    #         self.logger.log({'values/option_policy': wandb.Image(
                    #             os.path.join(self.path_to_store_logs,
                    #                          f'option_policy_{self.logger.id}_{self.episode}.png'))},
                    #                         step=self.episode)

                if ((((self.episode + 1) % log_every_episode) == 0) or (self.episode == 0)) and (
                        self.logger is not None):
                    self.animation = True
                    self.agent_pos.clear()
                # \\\logging\\\

                # Ad hoc terminal state
                self.agent.observe(obs, reward, is_first)
                self.current_action = self.agent.act()
                if self.logger is not None and self.task_complete:
                    self.log_task_complete()
                if self.early_stop:
                    break

                self.steps_cumulative += self.steps
                # print(self.total_reward)

                self.episode += 1
                self.steps = 0
                self.total_reward = 0

                if self.goal_reached:
                    self.on_new_goal()
                if self.task_complete:
                    self.on_new_task()
            else:
                self.steps += 1
                self.total_reward += reward

            self.agent.observe(obs, reward, is_first)
            self.current_action = self.agent.act()

            # ///logging///
            # if draw_options_stats:
            #     self.update_option_stats(self.environment.callmethod('is_terminal'))

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

        plt.imsave(
            os.path.join(
                self.path_to_store_logs,
                f'{logger.id}_episode_{episode}_step_{steps}.png'
            ), pic.astype('uint8')
        )
        plt.close()

    # def update_option_stats(self, is_terminal):
    #     option_block = self.agent.hierarchy.blocks[5]
    #     top_left_point = self.environment.env.renderer.shape.top_left_point
    #
    #     if option_block.made_decision and not is_terminal:
    #         current_option_id = option_block.current_option
    #         if self.current_option_id != current_option_id:
    #             if len(self.option_actions) != 0:
    #                 # update stats
    #                 self.option_end_pos = get_unshifted_pos(self.environment.env.agent.position,
    #                                                         top_left_point)
    #                 self.option_stat.update(self.current_option_id,
    #                                         self.option_start_pos,
    #                                         self.option_end_pos,
    #                                         self.option_actions,
    #                                         self.option_predicted_actions)
    #                 self.option_actions.clear()
    #                 self.option_predicted_actions = list()
    #
    #             self.option_start_pos = get_unshifted_pos(self.environment.env.agent.position,
    #                                                       top_left_point)
    #
    #         predicted_actions = list()
    #         if self.agent.hierarchy.output_block.predicted_options is not None:
    #             predicted_options = self.agent.hierarchy.output_block.sm.get_options_by_id(
    #                 self.agent.hierarchy.output_block.predicted_options)
    #             for o in predicted_options:
    #                 predicted_action_pattern = np.flatnonzero(o)
    #                 self.agent.muscles.set_active_input(predicted_action_pattern)
    #                 self.agent.muscles.depolarize_muscles()
    #                 action_pattern = self.agent.muscles.get_depolarized_muscles()
    #                 # convert muscles activation pattern to environment action
    #                 a = self.agent.action.get_action(action_pattern)
    #                 predicted_actions.append(a)
    #
    #             self.option_actions.append(self.current_action)
    #             self.option_predicted_actions.append(predicted_actions)
    #             self.current_option_id = current_option_id
    #     else:
    #         if len(self.option_actions) > 0:
    #             if option_block.current_option is not None:
    #                 last_option = option_block.current_option
    #             elif option_block.failed_option is not None:
    #                 last_option = option_block.failed_option
    #             elif option_block.completed_option is not None:
    #                 last_option = option_block.completed_option
    #             else:
    #                 last_option = None
    #             if last_option is not None:
    #                 last_option_id = last_option
    #                 self.option_end_pos = get_unshifted_pos(self.environment.env.agent.position,
    #                                                         top_left_point)
    #                 self.option_stat.update(last_option_id,
    #                                         self.option_start_pos,
    #                                         self.option_end_pos,
    #                                         self.option_actions,
    #                                         self.option_predicted_actions)
    #             self.option_actions.clear()
    #             self.option_predicted_actions = list()
    #             self.current_option_id = None

    # def get_options_usage_gain(self):
    #     options_usage_gain = dict()
    #     for id_, stats in self.option_stat.options.items():
    #         if id_ in self.last_options_usage.keys():
    #             last_value = self.last_options_usage[id_]
    #         else:
    #             last_value = 0
    #         options_usage_gain[id_] = stats['n_uses'] - last_value
    #     return options_usage_gain

    # def update_options_usage(self):
    #     last_options_usage = dict()
    #     for id_, stats in self.option_stat.options.items():
    #         last_options_usage[id_] = stats['n_uses']
    #     self.last_options_usage = last_options_usage

    def set_food_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.env.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        self.environment.env.modules['food'].generator.positions = positions

    def set_agent_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.env.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        self.environment.env.modules['agent'].positions = positions

    def set_pos_rand_rooms(
            self, agent_fixed_positions=None, food_fixed_positions=None, door_positions=None, wall_thickness=1
    ):
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
        plt.imsave(os.path.join(
            self.path_to_store_logs,
            f'map_{self.env_config["seed"]}_{self.episode}.png'),
            map_image.astype('uint8')
        )
        plt.close()
        logger.log({
                'maps/map': wandb.Image(os.path.join(
                    self.path_to_store_logs, f'map_{self.env_config["seed"]}_{self.episode}.png'
                ))
            },
            step=self.episode
        )

    def log_goal_complete(self):
        self.logger.log({
            'goal': self.total_terminals,
            'main_metrics/g_goal_steps': self.steps_per_goal,
            'main_metrics/g_task_steps': self.steps_per_task,
            'main_metrics/g_total_steps': self.steps_total,
            'main_metrics/g_episode': self.episode,
        }, step=self.episode)

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
        self.agent.flush_replay()

    @staticmethod
    def define_logging_metrics():
        wandb.define_metric("task")
        wandb.define_metric("main_metrics/steps_per_task", step_metric="task")
        wandb.define_metric("main_metrics/t_*", step_metric="task")

        wandb.define_metric("goal")
        wandb.define_metric("main_metrics/g_*", step_metric="goal")


def resolve_agent(name, **config):
    agent = None
    if name == 'dqn':
        from htm_rl.agents.dqn.agent import make_agent
        agent = make_agent(config)
    elif name == 'o-c':
        from htm_rl.agents.optcrit.agent import make_agent
        agent = make_agent(config)
    else:
        AttributeError(f'Unknown Deep RL agent {name}')

    return agent
