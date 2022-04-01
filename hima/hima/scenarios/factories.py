from hima.agents.agent import Agent
from hima.envs.env import Env
from hima.scenarios.config import FileConfig
from hima.scenarios.experiment import Experiment
from hima.scenarios.utils import filter_out_non_passable_items


def materialize_environment(env_config: dict, seed: int) -> Env:
    env_type = env_config['_type_']
    env_config = filter_out_non_passable_items(env_config, depth=2)
    if env_type == 'biogwlab':
        from hima.envs.biogwlab.env import BioGwLabEnvironment
        return BioGwLabEnvironment(seed=seed, **env_config)
    else:
        raise NameError(env_type)


def materialize_agent(agent_config: dict, seed: int, env: Env) -> Agent:
    agent_type = agent_config['_type_']
    agent_config = filter_out_non_passable_items(agent_config, depth=2)
    if agent_type == 'rnd':
        from hima.agents.rnd.agent import RndAgent
        return RndAgent(seed=seed, env=env)
    elif agent_type == 'q':
        from hima.agents.q.agent import QAgent
        return QAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'qmb':
        from hima.agents.qmb.agent import QModelBasedAgent
        return QModelBasedAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'dreamer':
        from hima.agents.dreamer.agent import DreamerAgent
        return DreamerAgent(seed=seed, env=env, **agent_config)
    else:
        raise NameError(agent_type)


def materialize_experiment(config: FileConfig) -> Experiment:
    scenario_type = config['_type_']
    if scenario_type == 'standard':
        from hima.scenarios.standard.experiment import StandardExperiment
        return StandardExperiment(config)
    elif scenario_type == 'dreaming condition':
        from hima.scenarios.dream_cond.experiment import DreamingConditionExperiment
        return DreamingConditionExperiment(config)
    else:
        raise NameError(scenario_type)


def inject_debugger(debug: dict, scenario, **kwargs):
    debugger_type = debug['_type_']

    if debugger_type == 'dreaming trajectory':
        from hima.agents.dreamer.debug.dreaming_trajectory_debugger import DreamingTrajectoryDebugger
        # noinspection PyUnusedLocal
        trajectory_debugger = DreamingTrajectoryDebugger(scenario, **kwargs)
    elif debugger_type == 'model':
        from hima.agents.qmb.debug.model_debugger import ModelDebugger
        # noinspection PyUnusedLocal
        model_debugger = ModelDebugger(scenario, **kwargs)
    elif debugger_type == 'encoding':
        from hima.agents.q.debug.encoding_debugger import EncodingDebugger
        # noinspection PyUnusedLocal
        encoding_debugger = EncodingDebugger(scenario, **kwargs)
    else:
        raise KeyError(debugger_type)
