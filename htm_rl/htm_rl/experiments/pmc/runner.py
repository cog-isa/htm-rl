from .environment import ReachAndGrasp2D
from .agent import BasicAgent


class Runner:
    def __init__(self, config, logger=None):
        self.environment = ReachAndGrasp2D(**config['environment'])
        self.agent = BasicAgent(**config['agent'])
        self.n_episodes = config['n_episodes']
        self.headless = config['headless']
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

                for sim_step in range(self.action_update_period//self.environment.time_constant):
                    self.environment.simulation_step()

    def run_gui(self):
        pass
