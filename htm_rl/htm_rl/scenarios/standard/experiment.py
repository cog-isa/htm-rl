import numpy as np
from tqdm import trange

from htm_rl.agents.agent import Agent
from htm_rl.agents.q.agent import QAgent
from htm_rl.agents.qmb.agent import QModelBasedAgent
from htm_rl.agents.rnd.agent import RndAgent
from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.dreamer0.agent import DreamerAgent as DreamerAgent0
from htm_rl.agents.ucb.agent import UcbAgent
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.env import Env
from htm_rl.scenarios.utils import ProgressPoint, filter_out_non_passable_items


class Experiment:
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

        self.env = self.materialize_environment(config['env'], config['env_seed'], config['envs'])
        self.agent = self.materialize_agent(config['agent'], config['agent_seed'], config['agents'], self.env)

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
            if first and self.progress.step > 0:
                break

            action = self.agent.act(reward, obs, first)
            self.env.act(action)

            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward

    @staticmethod
    def materialize_agent(name: str, seed: int, agent_configs: dict[str, dict], env: Env) -> Agent:
        agent_config: dict = agent_configs[name]

        agent_type = agent_config['_type_']
        agent_config = filter_out_non_passable_items(agent_config, depth=2)
        if agent_type == 'rnd':
            return RndAgent(seed=seed, env=env)
        elif agent_type == 'q':
            return QAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'qmb':
            return QModelBasedAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'ucb':
            return UcbAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'dreamer':
            return DreamerAgent(seed=seed, env=env, **agent_config)
        elif agent_type == 'dreamer0':
            return DreamerAgent0(seed=seed, env=env, **agent_config)
        else:
            raise NameError(agent_type)

    @staticmethod
    def materialize_environment(name: str, seed: int, env_configs: dict[str, dict]) -> Env:
        env_config: dict = env_configs[name]

        env_type = env_config['_type_']
        env_config = filter_out_non_passable_items(env_config, depth=2)
        if env_type == 'biogwlab':
            return BioGwLabEnvironment(seed=seed, **env_config)
        else:
            raise NameError(env_type)

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


class RunStats:
    steps: list[int]
    rewards: list[float]
    times: list[float]

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.times = []

    def append_stats(self, steps, total_reward, elapsed_time):
        self.steps.append(steps)
        self.rewards.append(total_reward)
        self.times.append(elapsed_time)

    def print_results(self):
        steps = np.array(self.steps)
        avg_len = steps.mean()
        last_10_pct = steps.shape[0] // 10
        last10_avg_len = steps[-last_10_pct:].mean()
        avg_reward = np.array(self.rewards).mean()
        avg_time = np.array(self.times).mean()
        elapsed = np.array(self.times).sum()
        print(
            f'Len10: {last10_avg_len: .2f}  Len: {avg_len: .2f}  '
            f'R: {avg_reward: .5f}  '
            f'EpT: {avg_time: .6f}  TotT: {elapsed: .6f}'
        )
