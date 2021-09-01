import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.common.utils import timed
from htm_rl.envs.env import Env
from htm_rl.scenarios.factories import materialize_environment, materialize_agent
from htm_rl.scenarios.standard.run_stats import RunStats
from htm_rl.scenarios.utils import ProgressPoint


class Scenario:
    config: dict

    n_episodes: int
    debug: dict
    debug_enabled: bool
    wandb: dict

    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(self, config: dict):
        self.config = config

        self.n_episodes = config['n_episodes']
        self.debug = config['debug']
        self.wandb = config['wandb']

        self.debug_enabled = self.debug['enabled']

        self.env = materialize_environment(config['envs'][config['env']], config['env_seed'])
        self.agent = materialize_agent(config['agents'][config['agent']], config['agent_seed'], self.env)

        self.progress = ProgressPoint()

    def run(self):
        train_stats = RunStats()
        wandb_run = self.init_wandb_run()

        if self.debug_enabled:
            from htm_rl.agents.qmb.debug.model_debugger import ModelDebugger
            print_images = self.debug['images']
            model_debugger = ModelDebugger(self, images=print_images)

        for _ in trange(self.n_episodes):
            (steps, reward), elapsed_time = self.run_episode()
            train_stats.append_stats(steps, reward, elapsed_time)
            self.progress.end_episode()

            if wandb_run is not None:
                wandb_run.log({
                    'steps': steps,
                    'reward': reward,
                    'elapsed_time': elapsed_time
                })

        if self.debug_enabled:
            anomalies = np.array(model_debugger.anomaly_tracker.anomalies)
            reward_anomalies = np.array(model_debugger.anomaly_tracker.reward_anomalies)
            print(round(anomalies.mean(), 4), round(reward_anomalies.mean(), 4))

        return train_stats

    @timed
    def run_episode(self):
        total_reward = 0.

        while True:
            reward, obs, first = self.env.observe()
            action = self.agent.act(reward, obs, first)
            if self.episode_ended(first):
                break
            self.env.act(action)

            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward

    def episode_ended(self, first: bool):
        return first and self.progress.step > 0

    def init_wandb_run(self):
        if not self.wandb['enabled']:
            return None

        import wandb
        project = self.wandb['project']
        assert project is not None, 'Wandb project name, set by `wandb.project` config field, is missing.'
        run = wandb.init(project=project, reinit=True, dir=self.config['base_dir'])
        run.config.agent = {
            'name': self.config['agent'],
            'type': self.agent.name,
            'seed': self.config['agent_seed']
        }
        run.config.environment = {
            'name': self.config['env'],
            'seed': self.config['env_seed']
        }
        return run
