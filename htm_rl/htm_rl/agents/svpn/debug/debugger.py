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
