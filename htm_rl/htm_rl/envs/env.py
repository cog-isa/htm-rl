from typing import Any

from htm_rl.common.sdr import SparseSdr


class Env:

    def observe(self) -> tuple[float, SparseSdr, bool]:
        raise NotImplementedError

    def get_info(self) -> dict:
        return dict()

    def act(self, action: Any) -> None:
        raise NotImplementedError

    @property
    def n_actions(self):
        raise NotImplementedError

    @property
    def output_sdr_size(self):
        raise NotImplementedError

    def callmethod(self, method: str, *args, **kwargs):
        return getattr(self, method)(*args, **kwargs)


class Wrapper(Env):
    env: Env
    root_env: Env

    def __init__(self, env):
        self.env = env
        self.root_env = unwrap(env)

    def observe(self) -> tuple[float, SparseSdr, bool]:
        return self.env.observe()

    def get_info(self) -> dict:
        return self.env.get_info()

    def act(self, action: Any) -> None:
        return self.env.act(action)

    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def output_sdr_size(self):
        return self.env.output_sdr_size

    def callmethod(
        self, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        return self.env.callmethod(method, *args, **kwargs)


def unwrap(env):
    if isinstance(env, Wrapper):
        env = env.root_env
    return env
