from inspect import getattr_static
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML, BaseLoader, SafeConstructor

from htm_rl.common.utils import isnone, ensure_list


class TagMethods:

    @staticmethod
    def generate_seeds(loader: BaseLoader, node):
        seed_gen_dict = loader.construct_mapping(node)
        base = seed_gen_dict['base']
        n_seeds = seed_gen_dict['n_seeds']
        seeds = np.random.default_rng(base).integers(0, 1_000_000, size=n_seeds)
        return seeds


def read_config(file_path: Path, verbose=False):
    yaml: YAML = YAML(typ='safe')
    register_classes(yaml)
    register_static_methods_as_tags(TagMethods, yaml)

    config = yaml.load(file_path)
    if verbose:
        from pprint import pprint
        pprint(config)
    return config


def register_classes(yaml: YAML):
    classes = [
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


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def __contains__(self, item):
        if not isinstance(item, str):
            return super(Config, self).__contains__(item)

        keys = item.split('.')
        if len(keys) == 1:
            return super(Config, self).__contains__(item)
        res = self
        for key in keys:
            if key not in res:
                return False
            res = res[key]
        return True

    def __getitem__(self, item):
        if not isinstance(item, str):
            return super(Config, self).__getitem__(item)

        keys = item.split('.')
        if len(keys) == 1:
            return super(Config, self).__getitem__(item)
        res = self
        for key in keys:
            res = res[key]
        return res

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            super(Config, self).__setitem__(key, value)
            return

        keys = key.split('.')
        if len(keys) == 1:
            super(Config, self).__setitem__(key, value)
            return

        res = self
        for k in keys[:-1]:
            if k not in res:
                res[k] = dict()
            res = res[k]
        res[keys[-1]] = value


class FileConfig(Config):
    path: Path
    content: dict

    _name: str

    def __init__(self, path: Path, name: str = None):
        self.path = path
        self.content = read_config(path)
        self._name = isnone(name, self.path.stem)

        super().__init__(self.content)

    @property
    def name(self):
        return self._name

    def read_subconfigs(self, key, prefix):
        self._ensure_dict(key)
        for name, val in self[key].items():
            if isinstance(val, dict):
                continue
            self.read_subconfig(f'{key}.{name}', prefix)

    def read_subconfig(self, key: str, prefix: str):
        if key not in self:
            return
        val: str = self[key]
        if isinstance(val, dict):
            return
        config_path = self.path.with_name(f'{prefix}_{val}.yml')
        name = key.split('.')[-1]
        config = FileConfig(config_path, name=name)

        self[key] = config

    def _ensure_dict(self, key):
        if not isinstance(self[key], dict):
            # has to be the name/names
            self[key] = ensure_list(self[key])
            d = dict()
            for name in self[key]:
                d[name] = name
            self[key] = d


class DictConfig(Config):
    content: dict

    _name: str

    def __init__(self, content: dict, name: str):
        self.content = content
        self._name = name

        super().__init__(self.content)

    @property
    def name(self):
        return self._name
