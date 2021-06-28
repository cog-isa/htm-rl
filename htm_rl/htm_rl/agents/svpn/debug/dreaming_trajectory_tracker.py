from htm_rl.agents.svpn.debug.providers import StateEncodingProvider
from htm_rl.agents.svpn.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.common.sdr import SparseSdr
from htm_rl.experiment import Experiment


class DreamingTrajectoryTracker(TrajectoryTracker):
    state_encoding_provider: StateEncodingProvider
    encoding_scheme: dict[tuple[int, int], SparseSdr]

    def __init__(self, experiment: Experiment):
        super().__init__(experiment, act_method_name='_move_in_dream')
        self.state_encoding_provider = StateEncodingProvider(experiment)
        self.encoding_scheme = dict()
        self.agent.set_breakpoint('_put_into_dream', self.on_act)

    def on_put_into_dreaming(self, agent, put_into_dream, *args, **kwargs):
        res = put_into_dream(*args, **kwargs)
        self.encoding_scheme = self.state_encoding_provider.get_encoding_scheme()
        return res

    def on_act(self, agent, act, *args, **kwargs):
        state = args[0]
        assert self.encoding_scheme

        position = self.state_encoding_provider.decode_state(state, self.encoding_scheme)
        assert position is not None
        self.heatmap[position] += 1
        return act(*args, **kwargs)