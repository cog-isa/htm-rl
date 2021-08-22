from htm_rl.agents.dreamer.agent import DreamerAgent
from htm_rl.agents.rnd.debug.debugger import Debugger
from htm_rl.agents.dreamer.debug.providers import ValueMapProvider
from htm_rl.agents.rnd.debug.env_map_provider import EnvMapProvider
from htm_rl.agents.rnd.debug.trajectory_tracker import TrajectoryTracker
from htm_rl.envs.biogwlab.environment import Environment

from htm_rl.scenarios.standard.experiment import Experiment
from htm_rl.scenarios.debug_output import ImageOutput


# noinspection PyUnresolvedReferences
class DreamingDebugger(Debugger):
    env: Environment
    agent: DreamerAgent

    waking: bool

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self.waking = False
        self.output_renderer = ImageOutput(experiment.config)
        self.env_map_provider = EnvMapProvider(experiment)
        self.trajectory_tracker = TrajectoryTracker(experiment)
        self.value_map_provider = ValueMapProvider(experiment)

        self.progress.set_breakpoint('end_episode', self.on_end_episode)
        self.agent.set_breakpoint('_wake', self.on_wake)
        self.agent.set_breakpoint('_reset_dreaming', self.on_reset_dreaming)

    def on_reset_dreaming(self, agent, func, *args, **kwargs):
        if self.waking:
            # waking from dreaming
            pass

        # starting new rollout

    def on_wake(self, agent, func, *args, **kwargs):
        self.waking = True
        func(*args, **kwargs)
        self.waking = False

    def on_end_episode(self, agent, func, *args, **kwargs):
        if self.output_renderer.is_empty:
            self._add_env_map()
            self._add_value_maps(q=True, v=True, greedy=True, exploration=True, ucb=True)
            self._add_trajectory()
            self.output_renderer.flush(
                f'end_episode_{self._default_config_identifier}_{self._default_progress_identifier}'
            )

        func(*args, **kwargs)

    def _add_trajectory(self):
        self.output_renderer.handle_img(
            self.trajectory_tracker.heatmap, self.trajectory_tracker.title, with_value_text=True
        )
        self.trajectory_tracker.reset()

    # noinspection PyUnboundLocalVariable
    def _add_value_maps(self, q: bool, v: bool, greedy: bool, exploration: bool, ucb: bool):
        self.value_map_provider.precompute(greedy or exploration, exploration or ucb)

        if greedy or exploration:
            Q = self.value_map_provider.Q
            V = self.value_map_provider.V(Q)
        if v and greedy:
            self.output_renderer.handle_img(V, 'V', with_value_text=True)
        if q and greedy:
            Q_render = self.value_map_provider.reshape_q_for_rendering(Q)
            self.output_renderer.handle_img(Q_render, 'Q', with_value_text=False)

        if ucb or exploration:
            UCB = self.value_map_provider.UCB

        if v and exploration:
            Q_exp = Q + UCB
            V_exp = self.value_map_provider.V(Q_exp)
            self.output_renderer.handle_img(V_exp, 'V exp', with_value_text=True)
        if q and exploration:
            Q_exp_render = self.value_map_provider.reshape_q_for_rendering(Q_exp)
            self.output_renderer.handle_img(Q_exp_render, 'Q exp', with_value_text=False)

        if ucb:
            UCB_render = self.value_map_provider.reshape_q_for_rendering(UCB)
            self.output_renderer.handle_img(UCB_render, 'UCB term', with_value_text=False)

    def _add_env_map(self):
        env_maps = self.env_map_provider.maps
        env_map_titles = self.env_map_provider.titles
        for i in range(1):
            self.output_renderer.handle_img(env_maps[i], env_map_titles[i])
