from hima.agents.agent import Agent
from hima.envs.env import Env, unwrap as env_unwrap
from hima.scenarios.standard.scenario import Scenario
from hima.scenarios.utils import ProgressPoint


class Debugger:
    scenario: Scenario
    env: Env
    progress: ProgressPoint

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.env = env_unwrap(scenario.env)
        self.progress = scenario.progress

    @property
    def agent(self) -> Agent:
        return self.scenario.agent

    @property
    def _default_env_identifier(self):
        config = self.scenario.config
        return f'{config["env_seed"]}'

    @property
    def _default_config_identifier(self) -> str:
        config = self.scenario.config
        return f'{config["agent"]}_{config["env_seed"]}_{config["agent_seed"]}'

    @property
    def _default_progress_identifier(self) -> str:
        if self.progress.is_new_episode:
            return f'{self.progress.episode}'
        return f'{self.progress.episode}_{self.progress.step}'

    @property
    def train_eval_mark(self):
        is_train = self.scenario.mode == 'train'
        train_eval_mark = 'A' if is_train else 'Z'
        return train_eval_mark

    def get_episode_filename(self, pp: ProgressPoint, train_eval_mark: str):
        config_id = self._default_config_identifier
        pp_id = f'{pp.episode}_{train_eval_mark}_{pp.step}'
        return f'end_episode_{config_id}_{pp_id}'
