from typing import Any, Dict, Tuple

from htm_rl.common.sdr import SparseSdr


class Env:

    def observe(self) -> Tuple[float, SparseSdr, bool]:
        raise NotImplementedError

    def get_info(self) -> Dict:
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
