from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.common.utils import timed
from htm_rl.envs.env import Env
from htm_rl.scenarios.factories import materialize_environment, materialize_agent, inject_debugger
from htm_rl.scenarios.standard.run_stats import RunStats
from htm_rl.scenarios.utils import ProgressPoint


class Scenario:
    config: dict

    n_episodes: int
    train_ep_before_eval: int

    debug: dict
    debug_enabled: bool
    wandb: dict

    mode: str
    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(
            self, config: dict, n_episodes: int, train_ep_before_eval: int, **_
    ):
        self.config = config
        self.n_episodes = n_episodes
        self.train_ep_before_eval = train_ep_before_eval

        self.debug = config['debug']
        self.debug_enabled = self.debug['enabled']

        self.wandb = config['wandb']

        self.mode = 'train'
        self.env = materialize_environment(config['envs'][config['env']], config['env_seed'])
        self.agent = materialize_agent(config['agents'][config['agent']], config['agent_seed'], self.env)
        self.progress = ProgressPoint()

    def run(self):
        train_stats = RunStats()
        eval_stats = RunStats()
        wandb_run = self.init_wandb_run()

        if self.debug_enabled:
            inject_debugger(self.debug, self, print_images=self.debug['print_images'])

        for _ in trange(self.n_episodes):
            self.run_episode_with_mode(
                train_stats, eval_stats, wandb_run
            )

        print(self.agent.dreamer._print_dreaming_stats())

        return train_stats, eval_stats

    def run_episode_with_mode(
            self, train_stats, eval_stats, wandb_run=None
    ):
        (steps, reward), elapsed_time = self.run_episode()
        train_stats.append_stats(steps, reward, elapsed_time)

        if self.should_eval:
            # self.progress.end_episode(increase_episode=False)
            # self.switch_to_state('eval')
            # (steps, reward), elapsed_time = self.run_episode()
            eval_stats.append_stats(steps, reward, elapsed_time)

            if wandb_run is not None:
                wandb_run.log({
                    'steps': steps,
                    'reward': reward,
                    'elapsed_time': elapsed_time
                })
            self.progress.end_episode()
            self.switch_to_state('train')
        else:
            self.progress.end_episode()

    @property
    def should_eval(self):
        return (self.progress.episode + 1) % self.train_ep_before_eval == 0

    def switch_to_state(self, new_state):
        if new_state == 'train':
            self.mode = 'train'
            self.agent.train = True
        elif new_state == 'eval':
            self.mode = 'eval'
            self.agent.train = False

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
