from inspect import getattr_static
from pathlib import Path

import numpy as np
from ruamel.yaml import BaseLoader, YAML, SafeConstructor


class TagMethods:

    @staticmethod
    def generate_seeds(loader: BaseLoader, node):
        seed_gen_dict = loader.construct_mapping(node)
        base = seed_gen_dict['base']
        n_seeds = seed_gen_dict['n_seeds']
        seeds = np.random.default_rng(base).integers(0, 1_000_000, size=n_seeds)
        seeds.sort()
        return seeds.tolist()


def read_config(file_path: Path, verbose=False):
    yaml: YAML = YAML(typ='safe')
    register_classes(yaml)
    register_static_methods_as_tags(TagMethods, yaml)

    config = yaml.load(file_path)
    if verbose:
        from pprint import pprint
        pprint(config)
    return config


def save_config(file_path: Path, config):
    from hima.scenarios.config import FileConfig
    config: FileConfig
    yaml: YAML = YAML(typ='unsafe')
    # the only option I found to prevent compact braces styles
    yaml.default_flow_style = False
    yaml.brace_single_entry_mapping_in_flow_sequence = False
    yaml.sort_base_mapping_type_on_output = False
    with file_path.open('w') as file:
        yaml.dump(config, file)


def register_classes(yaml: YAML):
    classes = [
    ]

    constructor: SafeConstructor = yaml.constructor
    constructor.deep_construct = True

    # noinspection PyShadowingNames
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