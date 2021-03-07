from typing import Tuple, Any, Dict

from htm_rl.common.sdr import SparseSdr
from htm_rl.envs.env import Env


class Wrapper(Env):
    env: Env

    def __init__(self, env):
        self.env = env

    def observe(self) -> Tuple[float, SparseSdr, bool]:
        return self.env.observe()

    def get_info(self) -> Dict:
        return self.env.get_info()

    def act(self, ac: Any) -> None:
        return self.env.act(ac)

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
    while isinstance(env, Wrapper):
        env = env.env
    return env
