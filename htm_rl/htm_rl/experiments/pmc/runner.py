import PIL.Image
from environment import ReachAndGrasp2D
from agent import BasicAgent
import pygame
import numpy as np
from utils import heatmap


class Runner:
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
                    probs = probs.reshape((20, int(probs.shape[1]**0.5), -1))
                    hm = heatmap(probs.mean(axis=0))
                    hm = hm.resize(self.sup_layout, PIL.Image.NEAREST)
                    image = pil_image_to_surface(hm)
                    screen.blit(image, dest=(self.sup_layout[0], self.sup_layout[1]))

            if update_screen:
                if self.environment.can_grab:
                    self.environment.reset()
                    self.agent.reset()
                    n_steps = 0
                self.environment.simulation_step()
                image = self.environment.render_rgb(
                    camera_resolution=self.camera_layout,
                    show_goal=show_goal)
                image = pil_image_to_surface(image)
                screen.blit(image, dest=(self.sup_layout[0]*2, 0))
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
                (f"update: {round(screen_update_period * screen_update_factor, 2)} ms   "   
                    f"r: {round(reward, 3)}   s: {n_steps}   ep: {n_episodes}"
                 f"  fps: {round(clock.get_fps())}"),
                False,
                (255, 255, 255))
            screen.blit(txt, (self.window_size[0] // 3, 0))

            pygame.display.flip()
            dt = clock.tick()
            screen_time += dt


if __name__ == '__main__':
    import yaml

    name = 'configs/base_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = Runner(config)
    runner.run()
