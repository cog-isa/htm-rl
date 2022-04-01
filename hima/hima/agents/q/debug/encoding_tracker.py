import numpy as np

from hima.agents.q.agent import QAgent
from hima.agents.q.sa_encoder import SaEncoder
from hima.agents.rnd.debug.agent_state_provider import AgentStateProvider
from hima.agents.rnd.debug.debugger import Debugger
from hima.common.debug import inject_debug_tools
from hima.common.utils import Coord2d, lin_sum
from hima.scenarios.debug_output import ImageOutput
from hima.scenarios.standard.scenario import Scenario


class EncodingTracker(Debugger):
    agent: QAgent

    position_provider: AgentStateProvider
    encoding_map: dict[Coord2d, set[int]]
    error: list[int]

    def __init__(self, scenario: Scenario):
        super(EncodingTracker, self).__init__(scenario)

        self.position_provider = AgentStateProvider(scenario)
        self.encoding_map = {}
        self.error = []

        sa_encoder = self.agent.sa_encoder
        inject_debug_tools(sa_encoder)
        # noinspection PyUnresolvedReferences
        sa_encoder.set_breakpoint('encode_state', self.on_encode_state)

    def on_encode_state(self, sa_encoder: SaEncoder, encode, *args, **kwargs):
        x = encode(*args, **kwargs)
        if not kwargs['learn']:
            return x

        new_x = set(x.tolist())
        pos = self.position_provider.position

        if pos in self.encoding_map:
            prev_x = self.encoding_map[pos]
            err = (len(new_x) - len(new_x & prev_x)) / len(x)
            prev_err = self.error[-1] if self.error else 0.
            err = lin_sum(prev_err, .04, err)
            self.error.append(err)

        self.encoding_map[pos] = new_x
        return x

    def print_plot(self, renderer: ImageOutput):
        error = np.array(self.error)
        renderer.handle_img(error, title='encoding error')
