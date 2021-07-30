from htm_rl.agents.agent import Agent, unwrap as agent_unwrap
from htm_rl.common.debug import inject_debug_tools
from htm_rl.envs.env import Env, unwrap as env_unwrap
from htm_rl.experiment import Experiment, ProgressPoint


class Debugger:
    experiment: Experiment
    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.env = env_unwrap(experiment.env)
        self.agent = agent_unwrap(experiment.agent)
        self.progress = experiment.progress

        inject_debug_tools(self.env)
        inject_debug_tools(self.agent)
        inject_debug_tools(self.progress)

    @property
    def _default_env_identifier(self):
        config = self.experiment.config
        return f'{config["env_seed"]}'

    @property
    def _default_config_identifier(self) -> str:
        config = self.experiment.config
        return f'{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}'

    @property
    def _default_progress_identifier(self) -> str:
        if self.progress.is_new_episode:
            return f'{self.progress.episode}'
        return f'{self.progress.episode}_{self.progress.step}'