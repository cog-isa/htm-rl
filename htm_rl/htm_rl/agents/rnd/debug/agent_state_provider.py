from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.agent import Agent as EnvAgent
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.scenarios.standard.experiment import Experiment


class AgentStateProvider(Debugger):
    env: Environment

    def __init__(self, experiment: Experiment):
        super(AgentStateProvider, self).__init__(experiment)
        self.origin = None

    @property
    def env_agent(self) -> EnvAgent:
        # noinspection PyTypeChecker
        return self.env.entities['agent']

    @property
    def state(self):
        return self.position, self.view_direction

    @property
    def position(self):
        return self.env_agent.position

    @property
    def view_direction(self):
        return self.env_agent.view_direction

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
        self.env_agent.position = isnone(position, self.env_agent.position)
        self.env_agent.view_direction = isnone(view_direction, self.env_agent.view_direction)