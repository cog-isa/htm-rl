import os.path
import pathlib
import random
from functools import partial

import imageio
import matplotlib.pyplot as plt
import numpy as np
import wandb

from htm_rl.agents.htm.htm_agent import Scenario
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.env import unwrap


class Runner:
    def __init__(self, config, logger=None):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)
        set_one_thread()

        if 'scenario' in config.keys():
            # noinspection PyTypeChecker
            self.scenario = Scenario(config['scenario'], self)
        else:
            self.scenario = None

        self.env_config = config['environment']
        self.environment: Environment = unwrap(BioGwLabEnvironment(**self.env_config))
        self.agent = resolve_agent(
            config['agent'],
            state_dim=self.environment.output_sdr_size,
            action_dim=self.environment.n_actions,
            **config['cagent']
        )

        self.total_terminals = 0
        self.logger = logger
        self.total_reward = 0
        self.animation = False
        self.level = -1
        self.task = 0
        self.goal = 0
        self.steps = 0
        self.steps_per_goal = 0
        self.steps_per_task = 0
        self.steps_total = 0
        self.steps_cumulative = 0
        self.all_steps = 0
        self.episode = 0
        self.current_action = None
        self.early_stop = False
        self.map_change_indicator = 0
        self.goal_reached = True
        self.task_complete = True
        self.last_terminal_stat = 0

        self.path_to_store_logs = config['path_to_store_logs']
        pathlib.Path(self.path_to_store_logs).mkdir(parents=True, exist_ok=True)
        self.get_path = partial(os.path.join, self.path_to_store_logs)

        self.seed = seed
        self.rng = random.Random(self.seed)

    def run_episodes(
            self, n_episodes, gif_every_episode, log_every_episode=1, animation_fps=3, **_
    ):
        self.total_reward = 0
        self.steps = 0
        self.steps_per_goal = 0
        self.steps_per_task = 0
        self.steps_total = 0
        self.level = -1
        self.episode = 0
        self.last_terminal_stat = -1
        self.total_terminals = -1
        self.goal = 0
        self.task = 0
        self.animation = False
        self.goal_reached = True
        self.task_complete = False
        log_enabled = self.logger is not None

        if log_enabled:
            self.define_logging_metrics()

        while True:
            reward, obs, is_first = self.environment.observe()

            if self.environment.is_terminal() and self.environment.items_collected > 0:
                # detect that the goal is reached
                self.goal_reached = True

            if is_first:
                self.finish_episode(log_enabled, log_every_episode, animation_fps)
                if self.task_complete:
                    # re-observe
                    reward, obs, is_first = self.environment.observe()
                self.init_new_episode(gif_every_episode)

                if self.episode > n_episodes or self.early_stop:
                    break

            self.total_reward += reward

            self.agent.observe(obs, reward, is_first)
            self.current_action = self.agent.act()

            if log_enabled and self.animation:
                self.draw_animation_frame(self.logger, self.episode, self.steps)

            self.environment.act(self.current_action)
            self.steps += 1

        if log_enabled:
            if self.episode > n_episodes:
                # if reached episodes limit, log the last stats unconditionally
                self.log_episode_complete()
                self.log_goal_complete()
                self.log_task_complete()

            self.logger.log({"total_steps": self.steps_total}, step=self.episode)

    def finish_episode(self, log_enabled, log_schedule, animation_fps):
        self.steps_per_goal += self.steps
        self.steps_per_task += self.steps
        self.steps_total += self.steps
        self.all_steps += self.steps
        self.steps_cumulative += self.steps

        if self.goal_reached:
            self.total_terminals += 1
            self.last_terminal_stat += 1

        if self.scenario is not None:
            # re-resets environment if needed
            self.scenario.check_conditions()

        self.map_change_indicator = 1 if self.task_complete and self.episode > 0 else 0

        log_now = self.episode % log_schedule == 0
        log_now = log_now or self.animation or self.goal_reached or self.task_complete

        if log_enabled and log_now:
            if self.animation:
                # log all saved frames for the finished episode
                self.animation = False
                self.log_gif_animation(animation_fps)

            self.log_episode_complete()

            if self.goal_reached:
                self.log_goal_complete()
            if self.task_complete:
                self.log_task_complete()

    def init_new_episode(self, gif_schedule):
        self.episode += 1
        self.steps = 0
        self.total_reward = 0

        if self.goal_reached:
            self.goal_reached = False
            self.steps_per_goal = 0
            self.goal += 1
            self.draw_options_debug()

        if self.task_complete:
            self.task_complete = False
            self.steps_per_task = 0
            self.task += 1
            self.agent.flush_replay()

        if gif_schedule > 0 and (self.episode == 1 or self.episode % gif_schedule == 0):
            self.animation = True

    # FIXME: HACKS HACKS HACKS
    def draw_options_debug(self):
        from pathlib import Path
        import torch
        from numpy import ma
        from htm_rl.agents.optcrit.agent import OptionCriticAgent
        from htm_rl.common.utils import softmax
        from htm_rl.agents.dqn.network import to_np
        # 99-th goal
        if (self.goal + 1) % 100 != 0 or not isinstance(self.agent, OptionCriticAgent):
            return

        # if self.logger is not None:
        #     self.draw_map(self.logger)

        def get_observations():
            from htm_rl.envs.biogwlab.module import EntityType
            height, width = self.environment.shape
            obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]
            position_provider = AgentStateProvider(self.environment)
            encoding_scheme = {}

            for i in range(height):
                for j in range(width):
                    if obstacle_mask[i, j]:
                        continue
                    position = i, j
                    position_provider.overwrite(position)
                    obs = self.environment.render()
                    encoding_scheme[position] = obs

            position_provider.restore()
            return encoding_scheme

        observations = get_observations()
        shape = self.environment.shape + (self.environment.n_actions,)
        op_policy = ma.masked_all(shape, dtype=np.float)
        op_beta = ma.masked_all(shape, dtype=np.float)

        for position, obs in observations.items():
            with torch.no_grad():
                obs = self.agent.to_dense(obs)
                prediction = self.agent.network(obs)

            op_probs = softmax(to_np(prediction['q']), self.agent.config.cr_softmax_temp)
            op_policy[position] = op_probs

            beta_probs = to_np(prediction['beta'])
            op_beta[position] = beta_probs

        op_policy = op_policy.filled(.0)
        op_policy[0, 0, 0] = 1.
        op_policy[0, 0, 1] = .5
        op_policy[0, 0, 2] = .25
        op_beta = op_beta.filled(.0)
        op_beta[0, 0, 0] = 1.
        op_beta[0, 0, 1] = .5
        op_beta[0, 0, 2] = .25

        from htm_rl.common.plot_utils import plot_grid_images
        op_policy_path = Path(os.path.join(self.path_to_store_logs, f'op_policy_{self.task}.png'))
        op_beta_path = Path(os.path.join(self.path_to_store_logs, f'op_beta_{self.task}.png'))

        show = False
        # plot_grid_images(images=op_policy, titles='op_policy', show=show, save_path=op_policy_path)
        # plot_grid_images(images=op_beta, titles='op_beta', show=show, save_path=op_beta_path)

        plot_grid_images(
            images=[op_policy, op_beta],
            titles=['op_policy', 'op_beta'],
            show=show,
            save_path=op_policy_path
        )

        # print(str(op_policy_path))
        if self.logger is not None:
            self.logger.log({'maps/t_op_policy': wandb.Image(str(op_policy_path))}, step=self.episode)
            # self.logger.log({'maps/t_op_beta': wandb.Image(str(op_beta_path))}, step=self.episode)

    def log_gif_animation(self, animation_fps):
        # TODO: replace with in-memory storage
        with imageio.get_writer(
                self.get_path(f'{self.logger.id}_episode_{self.episode}.gif'),
                mode='I',
                fps=animation_fps
        ) as writer:
            for i in range(self.steps):
                image = imageio.imread(
                    self.get_path(
                        f'{self.logger.id}_episode_{self.episode}_step_{i}.png'
                    )
                )
                writer.append_data(image)
        # noinspection PyTypeChecker
        gif_video = wandb.Video(
            self.get_path(f'{self.logger.id}_episode_{self.episode}.gif'),
            fps=animation_fps,
            format='gif'
        )
        self.logger.log({f'behavior_samples/animation': gif_video}, step=self.episode)

    def draw_animation_frame(self, logger, episode, steps):
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

    def set_food_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        # noinspection PyUnresolvedReferences
        self.environment.modules['food'].generator.positions = positions

    def set_agent_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        positions = [self.environment.renderer.shape.shift_relative_to_corner(pos) for pos in positions]
        # noinspection PyUnresolvedReferences
        self.environment.modules['agent'].positions = positions

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

        # noinspection PyShadowingNames
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

    def log_episode_complete(self):
        self.logger.log(
            {
                'main_metrics/steps': self.steps,
                'reward': self.total_reward,
                'episode': self.episode,
                'main_metrics/level': self.level,
                'main_metrics/total_terminals': self.goal,
                'main_metrics/steps_cumulative': self.steps_cumulative,
                'main_metrics/total_steps': self.steps_total,
                'main_metrics/map_change_indicator': self.map_change_indicator,
                'main_metrics/all_steps': self.all_steps,
            },
            step=self.episode
        )

    def log_goal_complete(self):
        self.logger.log(
            {
                'goal': self.goal,
                'main_metrics/g_goal_steps': self.steps_per_goal,
                'main_metrics/g_task_steps': self.steps_per_task,
                'main_metrics/g_total_steps': self.steps_total,
                'main_metrics/g_episode': self.episode,
            },
            step=self.episode
        )

    def log_task_complete(self):
        self.logger.log(
            {
                'task': self.task,
                'main_metrics/steps_per_task': self.steps_per_task,
                'main_metrics/t_task_steps': self.steps_per_task,
                'main_metrics/t_total_steps': self.steps_total
            },
            step=self.episode
        )

    @staticmethod
    def define_logging_metrics():
        wandb.define_metric("task")
        wandb.define_metric("main_metrics/steps_per_task", step_metric="task")
        wandb.define_metric("main_metrics/t_*", step_metric="task")
        # wandb.define_metric("maps/t_*", step_metric="task")

        wandb.define_metric("goal")
        wandb.define_metric("main_metrics/g_*", step_metric="goal")


def set_one_thread():
    import torch

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


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


class AgentStateProvider:
    env: Environment

    def __init__(self, env: Environment):
        self.env = env
        self.origin = None

    @property
    def state(self):
        return self.position, self.view_direction

    @property
    def position(self):
        return self.env.agent.position

    @property
    def view_direction(self):
        return self.env.agent.view_direction

    def overwrite(self, position=None, view_direction=None):
        if self.origin is None:
            self.origin = self.state
        self._set(position, view_direction)

    def restore(self):
        if self.origin is None:
            raise ValueError('Nothing to restore')
        self._set(*self.origin)
        self.origin = None

    def _set(self, position, view_direction):
        from htm_rl.common.utils import isnone

        self.env.agent.position = isnone(position, self.env.agent.position)
        self.env.agent.view_direction = isnone(view_direction, self.env.agent.view_direction)