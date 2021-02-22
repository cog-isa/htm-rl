import random
from inspect import getattr_static
from pathlib import Path
from pprint import pprint

import numpy as np
from ruamel.yaml import YAML, BaseLoader, SafeConstructor

from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.agents.ucb.ucb_actor_critic import UcbActorCritic
from htm_rl.agents.ucb.ucb_agent import UcbAgent
from htm_rl.agents.ucb.ucb_experiment_runner import UcbExperimentRunner
from htm_rl.common.ucb_encoders import UcbIntBucketEncoder, UcbSdrConcatenator
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator
from htm_rl.envs.gridworld_map_generator import GridworldMapGenerator
from htm_rl.htm_plugins.temporal_memory import TemporalMemory
from htm_rl.htm_plugins.ucb_spatial_pooler import UcbSpatialPooler


class RandomSeedSetter:
    seed: int

    def __init__(self, seed):
        self.seed = seed
        self.reset()

    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)


def read_config(file_path: Path, verbose=False):
    yaml: YAML = YAML(typ='safe')
    register_classes(yaml)
    config = yaml.load(file_path)
    if verbose:
        pprint(config)
    return config


def register_classes(yaml: YAML):
    classes = [
        RandomSeedSetter,
        TemporalMemory,
        GridworldMapGenerator,
        RunResultsProcessor,
        BioGwLabEnvGenerator,
        UcbExperimentRunner, UcbAgent, UcbActorCritic,
        UcbSpatialPooler,
        UcbIntBucketEncoder, UcbSdrConcatenator,
    ]

    constructor: SafeConstructor = yaml.constructor
    constructor.deep_construct = True

    def create_class_factory(cls):
        def construct_and_init_object(loader: BaseLoader, node):
            kwargs = loader.construct_mapping(node, deep=True)
            return cls(**kwargs)

        return construct_and_init_object

    for cls in classes:
        tag = f'!{cls.__name__}'
        constructor.add_constructor(tag, create_class_factory(cls))


def register_static_methods_as_tags(cls, yaml: YAML):
    constructor: SafeConstructor = yaml.constructor
    constructor.deep_construct = True

    all_funcs = [
        (func, getattr_static(cls, func))
        for func in dir(cls)
    ]
    funcs = [
        (tag, getattr(cls, tag))
        for tag, func in all_funcs
        if isinstance(func, staticmethod)
    ]
    for tag, func in funcs:
        constructor.add_constructor(f'!{tag}', func)



