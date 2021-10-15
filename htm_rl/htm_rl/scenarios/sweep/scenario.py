import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.common.utils import timed, isnone
from htm_rl.envs.env import Env
from htm_rl.scenarios.factories import materialize_environment, materialize_agent, inject_debugger
from htm_rl.scenarios.standard.run_stats import RunStats
from htm_rl.scenarios.utils import ProgressPoint


class Scenario:
    config: dict

    n_episodes: int
    episodes_per_task: int
    steps_per_task: int
    total_steps: int

    train_ep_before_eval: int

    debug: dict
    debug_enabled: bool

    mode: str
    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(
            self, config: dict, n_episodes: int, train_ep_before_eval: int,
            progress: ProgressPoint = None, total_steps: int = 0,
            **_
    ):
        self.config = config
        self.n_episodes = n_episodes
        self.train_ep_before_eval = train_ep_before_eval

        self.debug = config['debug']
        self.debug_enabled = self.debug['enabled']

        self.mode = 'train'
        self.env = materialize_environment(config['envs'][config['env']], config['env_seed'])
        self.agent = materialize_agent(config['agents'][config['agent']], config['agent_seed'], self.env)
        self.progress = isnone(progress, ProgressPoint())
        self.total_steps = total_steps

    def run(self, wandb_run):
        train_stats = RunStats()

        if self.debug_enabled:
            inject_debugger(self.debug, self, print_images=self.debug['print_images'])

        for _ in trange(self.n_episodes):
            self.run_episode_with_mode(
                train_stats, wandb_run
            )

        # if wandb_run is not None:
        #     wandb_run.finish()
        return train_stats

    def run_episode_with_mode(
            self, train_stats, wandb_run=None
    ):
        (steps, reward), elapsed_time = self.run_episode()
        train_stats.append_stats(steps, reward, elapsed_time)
        self.total_steps += steps

        if self.should_eval:
            self.log_eval(wandb_run, train_stats)

        self.progress.end_episode()

    def log_eval(self, wandb_run, train_stats: RunStats):
        if wandb_run is None:
            self.agent.dreamer._print_dreaming_stats()
            return

        eval_steps = self.train_ep_before_eval
        wandb_run.log({
            'total_steps': self.total_steps,
            'steps': np.mean(train_stats.steps[-eval_steps:]),
            'reward': np.mean(train_stats.rewards[-eval_steps:]),
            'elapsed_time': np.mean(train_stats.times[-eval_steps:]),
        }, step=self.progress.episode)

        self.try_log_dreamer(wandb_run)

    def try_log_dreamer(self, wandb_run, with_reset=True):
        from htm_rl.agents.dreamer.agent import DreamerAgent
        if isinstance(self.agent, DreamerAgent):
            self.agent.dreamer.log_stats(wandb_run, self.progress.episode, with_reset)

    @property
    def should_eval(self):
        return (self.progress.episode + 1) % self.train_ep_before_eval == 0

    @staticmethod
    def update_progress(pbar=None):
        if pbar is not None:
            pbar.update(1)

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
