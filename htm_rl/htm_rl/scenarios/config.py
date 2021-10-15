from pathlib import Path

import numpy as np

from htm_rl.common.utils import isnone, ensure_list
from htm_rl.scenarios.yaml_utils import read_config, save_config


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
            if not key:
                # allows additional dots that are skipped
                continue
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
            if not k:
                # allows additional dots that are skipped
                continue
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
        self[key] = self._ensure_dict(self[key])
        for name, val in self[key].items():
            if isinstance(val, dict):
                continue
            self.read_subconfig(f'{key}.{name}', prefix)

    def read_subconfig(self, key: str, prefix: str):
        if key not in self:
            return
        val: str = self[key]
        if val is None or isinstance(val, dict):
            return
        config_path = self.path.with_name(f'{prefix}_{val}.yml')
        name = key.split('.')[-1]
        config = FileConfig(config_path, name=name)

        self[key] = config

    def autosave(self):
        autosave_dir: Path = self['autosave_dir']
        rng = np.random.default_rng()
        for _ in range(10):
            name = f'_autosave_{rng.integers(10000):05d}.yml'
            autosave = autosave_dir.joinpath(name)
            if not autosave.exists():
                save_config(autosave, self.as_dict())
                print(f'Autosave config to {autosave}')
                break

    @staticmethod
    def _ensure_dict(val) -> dict:
        if not isinstance(val, dict):
            val = ensure_list(val)
            d = dict()
            for name in val:
                d[name] = name
            return d
        return val

    def as_dict(self):
        def _as_dict(d):
            res = {}
            for k, v in d.items():
                if isinstance(v, FileConfig):
                    res[k] = v.as_dict()
                elif isinstance(v, dict):
                    res[k] = _as_dict(v)
                elif isinstance(v, Path):
                    res[k] = str(v)
                else:
                    res[k] = v
            return res

        result = _as_dict(self)
        result.update({
            'name': self.name, 'path': str(self.path)
        })
        return result


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
