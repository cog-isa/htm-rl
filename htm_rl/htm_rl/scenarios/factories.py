from htm_rl.agents.agent import Agent
from htm_rl.envs.env import Env
from htm_rl.scenarios.config import FileConfig
from htm_rl.scenarios.experiment import Experiment
from htm_rl.scenarios.utils import filter_out_non_passable_items


def materialize_environment(env_config: dict, seed: int) -> Env:
    env_type = env_config['_type_']
    env_config = filter_out_non_passable_items(env_config, depth=2)
    if env_type == 'biogwlab':
        from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
        return BioGwLabEnvironment(seed=seed, **env_config)
    else:
        raise NameError(env_type)


def materialize_agent(agent_config: dict, seed: int, env: Env) -> Agent:
    agent_type = agent_config['_type_']
    agent_config = filter_out_non_passable_items(agent_config, depth=2)
    if agent_type == 'rnd':
        from htm_rl.agents.rnd.agent import RndAgent
        return RndAgent(seed=seed, env=env)
    elif agent_type == 'q':
        from htm_rl.agents.q.agent import QAgent
        return QAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'qmb':
        from htm_rl.agents.qmb.agent import QModelBasedAgent
        return QModelBasedAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'ucb':
        from htm_rl.agents.ucb.agent import UcbAgent
        return UcbAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'dreamer':
        from htm_rl.agents.dreamer.agent import DreamerAgent
        return DreamerAgent(seed=seed, env=env, **agent_config)
    elif agent_type == 'dreamer0':
        from htm_rl.agents.dreamer0.agent import DreamerAgent as DreamerAgent0
        return DreamerAgent0(seed=seed, env=env, **agent_config)
    else:
        raise NameError(agent_type)


def materialize_experiment(config: FileConfig) -> Experiment:
    scenario_type = config['_type_']
    if scenario_type == 'standard':
        from htm_rl.scenarios.standard.experiment import StandardExperiment
        return StandardExperiment(config)
    elif scenario_type == 'dreaming condition':
        from htm_rl.scenarios.dream_cond.experiment import DreamingConditionExperiment
        return DreamingConditionExperiment(config)
    else:
        raise NameError(scenario_type)
