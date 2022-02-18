import PIL.Image
from environment import ReachAndGrasp2D
from htm_rl.envs.coppelia.environment import ArmEnv
from agent import BasicAgent
import pygame
import numpy as np
from utils import heatmap
from htm_rl.common.scenario import Scenario
import imageio
import os
import wandb
import matplotlib.pyplot as plt
from adapters import ActionAdapter


class RunnerRAG2D:
    def __init__(self, config, logger=None):
        self.environment = ReachAndGrasp2D(**config['environment'])

        self.agent = BasicAgent(self.environment.camera_resolution,
                                **config['agent'])

        self.n_episodes = config['n_episodes']
        self.headless = config['headless']
        if not self.headless:
            if 'window_size' in config.keys():
                self.window_size = tuple(config['window_size'])
            else:
                self.window_size = (600, 400)
        else:
            self.window_size = None

        self.camera_layout = (self.window_size[0] // 2, self.window_size[1])
        self.sup_layout = (self.window_size[0] // 4, self.window_size[1] // 2)

        self.action_update_period = config['action_update_period']
        self.max_steps = config['max_steps']

    def run(self):
        if self.headless:
            self.run_headless()
        else:
            self.run_gui()

    def run_headless(self):
        for episode in range(self.n_episodes):
            for step in range(self.max_steps):
                reward, obs = self.environment.obs()

                action = self.agent.make_action(obs)
                self.agent.reinforce(reward)

                self.environment.act(action)

                for sim_step in range(self.action_update_period // self.environment.time_constant):
                    self.environment.simulation_step()

    def run_gui(self):
        def pil_image_to_surface(pil_image):
            pil_image = pil_image.convert('RGB')
            return pygame.image.fromstring(
                pil_image.tobytes(), pil_image.size, pil_image.mode).convert()

        # initialize the pygame module
        pygame.init()
        pygame.font.init()
        font = pygame.font.SysFont('arial', 14)
        # load and set the logo
        # logo = pygame.image.load("logo32x32.png")
        # pygame.display.set_icon(logo)
        pygame.display.set_caption("Reach And Grasp")

        # create a surface on screen that has the size of 240 x 180
        screen = pygame.display.set_mode(self.window_size)
        clock = pygame.time.Clock()
        # controls
        key_actions = (
            (pygame.K_LEFT, "slowdown"),
            (pygame.K_RIGHT, "speedup"),
            (pygame.K_r, "reset"),
            (pygame.K_t, "show_target"),
            (pygame.K_m, "update_map"),
            (pygame.K_a, "show_activation"),
            (pygame.K_c, "update_connections"),
            (pygame.K_s, "show_stimulus"))
        # define a variable to control the main loop
        running = True

        # main loop
        n_steps = 0
        n_episodes = 0
        screen_time = 0
        action_time = 0
        screen_update_period = self.environment.time_constant
        action_update_period = self.action_update_period // self.environment.time_constant
        show_goal = False
        show_activation = 0
        show_stimulus = False
        reward = 0
        screen_update_factor = 1
        while running:
            update_connections = False
            update_map = False
            # event handling, gets all event from the event queue
            action = None
            for event in pygame.event.get():
                # only do something if the event is of type QUIT
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    running = False
                if event.type == pygame.KEYDOWN:
                    for k, a in key_actions:
                        if event.key == k:
                            action = a
            if action == 'reset':
                self.environment.reset()
                self.agent.reset()
                n_steps = 0
            elif action == 'speedup':
                screen_update_factor -= 0.1
                screen_update_factor = max(screen_update_factor, 0)
            elif action == 'slowdown':
                screen_update_factor += 0.1
            elif action == 'show_target':
                show_goal = not show_goal
            elif action == 'update_map':
                update_map = True
            elif action == 'show_activation':
                show_activation += 1
                show_activation %= 3
            elif action == 'update_connections':
                update_connections = True
            elif action == 'show_stimulus':
                show_stimulus = not show_stimulus

            update_screen = False
            update_action = False
            if screen_time >= screen_update_period * screen_update_factor:
                update_screen = True
                screen_time = 0
            if action_time >= action_update_period:
                update_action = True
                action_time = 0

            if update_action:
                reward, obs = self.environment.obs()
                action = self.agent.make_action(obs)
                self.agent.reinforce(reward)
                self.environment.act(action)
                if n_steps == 0:
                    n_episodes += 1
                n_steps += 1
                if show_activation:
                    im_size = int(round(self.agent.pmc.n_neurons ** 0.5) + 1)
                    probs = np.zeros(im_size ** 2)
                    if show_activation == 1:
                        probs[self.agent.response] = self.agent.probs
                    elif show_activation == 2:
                        probs[self.agent.response] = 1
                    hm = heatmap(probs.reshape((im_size, im_size)))
                    hm = hm.resize(self.sup_layout, PIL.Image.NEAREST)
                    image = pil_image_to_surface(hm)
                    screen.blit(image, dest=(self.sup_layout[0], 0))
                if show_stimulus:
                    probs = np.zeros(self.agent.v1.output_sdr_size)
                    probs[self.agent.bg.stri.current_stimulus] = 1
                    probs = probs.reshape((20, -1))
                    probs = probs.reshape((20, int(probs.shape[1] ** 0.5), -1))
                    hm = heatmap(probs.mean(axis=0))
                    hm = hm.resize(self.sup_layout, PIL.Image.NEAREST)
                    image = pil_image_to_surface(hm)
                    screen.blit(image, dest=(self.sup_layout[0], self.sup_layout[1]))

            if update_screen:
                if self.environment.can_grab:
                    reward, obs = self.environment.obs()
                    action = self.agent.make_action(obs)
                    self.agent.reinforce(reward)
                    self.environment.reset()
                    self.agent.reset()
                    n_steps = 0
                self.environment.simulation_step()
                image = self.environment.render_rgb(
                    camera_resolution=self.camera_layout,
                    show_goal=show_goal)
                image = pil_image_to_surface(image)
                screen.blit(image, dest=(self.sup_layout[0] * 2, 0))
                action_time += 1
            if update_map:
                hm = self.agent.pmc.distance_matrix_heatmap()
                hm = hm.resize(self.sup_layout, PIL.Image.NEAREST)
                image = pil_image_to_surface(hm)
                screen.blit(image, dest=(0, 0))
            if update_connections:
                hm = heatmap(
                    (self.agent.pmc.connections > self.agent.pmc.connected_threshold).astype(float)
                )
                hm = hm.resize(self.sup_layout, PIL.Image.NEAREST)
                image = pil_image_to_surface(hm)
                screen.blit(image, dest=(0, self.sup_layout[1]))

            if n_steps > self.max_steps:
                self.environment.reset()
                self.agent.reset()
                n_steps = 0
            # update text info
            txt = font.render(
                "update: {:n} ms r: {:.3f}   s: {:n}   ep: {:n} fps: {:.0f}".format(
                    screen_update_period * screen_update_factor,
                    reward,
                    n_steps,
                    n_episodes,
                    clock.get_fps()),
                False,
                (255, 255, 255))
            screen.blit(txt, (self.window_size[0] // 3, 0))

            pygame.display.flip()
            dt = clock.tick()
            screen_time += dt


class RunnerArm:
    def __init__(self, config, logger=None):
        self.log_buffer = dict()
        self.logger = logger
        self.shuffle_tasks = config['shuffle_tasks']
        self.n_tasks = config['n_tasks']
        self.path_to_store_logs = config['path_to_store_logs']
        self.recording_fps = config['recording_fps']
        self.limits = config['workspace_limits']
        # optional
        self.goals = config.get('goals')

        self.running = True
        self.n_terminals = 0
        self.step = 0
        self.episode = 0
        self.task = None
        self.total_steps = 0
        self.total_episodes = 0
        self.episodes_per_epoch = 0
        self.task_n = -1
        self.task_id = None
        self.epoch = 0

        self.capture_frames = False

        self.environment = ArmEnv(**config['environment'])

        self.agent = BasicAgent(self.environment.camera.get_resolution(),
                                **config['agent'])

        self.action_adapter = ActionAdapter(self.limits)

        self.scenario = Scenario(config['scenario'], self)

        self.seed = config.get('seed')
        self._rng = np.random.default_rng(self.seed)

        self.tasks = None
        self.tasks_ids = np.arange(self.n_tasks)
        # min and max values for every axis
        if self.goals is None:
            self.goals = self.generate_goals()

        if self.goals.shape[0] != self.n_tasks:
            self.tasks = self.goals[self._rng.choice(np.arange(self.goals.shape[0]), self.n_tasks)]
        else:
            self.tasks = self.goals

    def run(self):
        self.next_task()
        while True:
            self.scenario.check_conditions()
            if not self.running:
                self.environment.shutdown()
                break

            reward, obs, is_first = self.environment.observe()

            if self.logger is not None:
                self.log(is_first)

            if is_first:
                if self.step > 0:
                    self.n_terminals += 1
                self.step = 0
                self.episode += 1
                self.total_episodes += 1
                self.episodes_per_epoch += 1
                self.agent.reset()
            else:
                self.step += 1
                self.total_steps += 1

            action = self.agent.make_action(obs[0])
            action = self.action_adapter.adapt(action)
            self.agent.reinforce(reward)
            self.environment.act(action)

    def next_task(self):
        self.n_terminals = 0
        self.log_buffer['episodes_per_task'] = self.episode
        self.episode = 0
        self.task_n += 1

        if self.task_n >= self.n_tasks:
            self.task_n = 0
            self.epoch += 1
            self.log_buffer['episodes_per_epoch'] = self.episodes_per_epoch
            self.episodes_per_epoch = 0
            if self.shuffle_tasks:
                self._rng.shuffle(self.tasks_ids)

        self.task_id = self.tasks_ids[self.task_n]
        self.task = self.tasks[self.task_id]

        self.env_reset()
        self.environment.set_target_position(self.task)

    def stop(self):
        self.running = False

    def env_reset(self):
        self.environment.reset()

    def record(self):
        self.capture_frames = True

    def log(self, is_first):
        if is_first:
            if self.total_episodes > 0:
                self.logger.log(
                    {'main_metrics/steps': self.step, 'episode': self.total_episodes,
                     'main_metrics/task_id': self.task_id,
                     'main_metrics/total_steps': self.total_steps,
                     },
                    step=self.total_episodes)

            if self.episode == 0 and self.total_episodes > 0:
                self.logger.log(
                    {'main_metrics/episodes_per_task': self.log_buffer['episodes_per_task']},
                    step=self.task_id
                )
            if self.episodes_per_epoch == 0 and self.epoch > 0:
                self.logger.log(
                    {'main_metrics/episodes_per_epoch': self.log_buffer['episodes_per_epoch']},
                    step=self.epoch
                )

            if self.capture_frames:
                # log all saved frames for this episode
                self.capture_frames = False
                with imageio.get_writer(os.path.join(self.path_to_store_logs,
                                                     f'{self.logger.id}_episode_{self.total_episodes}.gif'),
                                        mode='I',
                                        fps=self.recording_fps) as writer:
                    for i in range(self.step):
                        image = imageio.imread(os.path.join(self.path_to_store_logs,
                                                            f'{self.logger.id}_episode_{self.total_episodes}_step_{i}.png'))
                        writer.append_data(image)
                self.logger.log(
                    {f'behavior_samples/animation': wandb.Video(
                        os.path.join(self.path_to_store_logs,
                                     f'{self.logger.id}_episode_{self.total_episodes}.gif'),
                        fps=self.recording_fps,
                        format='gif')}, step=self.total_episodes)
        else:
            if self.capture_frames:
                self.draw_animation_frame()

    def draw_animation_frame(self):
        pic = self.environment.camera.capture_rgb() * 255
        plt.imsave(os.path.join(self.path_to_store_logs,
                                f'{self.logger.id}_episode_{self.total_episodes}_step_{self.step}.png'), pic.astype('uint8'))
        plt.close()

    def generate_goals(self):
        r = self._rng.uniform(self.limits['r'][0], self.limits['r'][1], self.n_tasks)
        phi = self._rng.uniform(self.limits['phi'][0], self.limits['phi'][1], self.n_tasks)
        phi = np.radians(phi)
        h = self._rng.uniform(self.limits['h'][0], self.limits['h'][1], self.n_tasks)

        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return np.vstack([x, y, h]).T


