from collections import deque
from itertools import islice
from typing import Any, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm

from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import timed, trace
from htm_rl.envs.gridworld_map_generator import GridworldMapGenerator
from htm_rl.envs.gridworld_mdp import GridworldMdp
from htm_rl.envs.mdp import Mdp


class Agent:
    memory: Memory
    planner: Planner
    _n_actions: int
    _use_cooldown: bool
    _planning_horizon: int

    def __init__(self, memory: Memory, planner: Planner, n_actions, use_cooldown=False):
        self.memory = memory
        self.planner = planner
        self._n_actions = n_actions
        self._use_cooldown = use_cooldown
        self._planning_horizon = self.planner.planning_horizon
        self.set_planning_horizon(self._planning_horizon)
        self._goal_state = None

    def set_planning_horizon(self, planning_horizon: int):
        self.planner.planning_horizon = planning_horizon
        self._planning_horizon = planning_horizon

        self._planning_enabled = planning_horizon > 0
        self._n_times_planned = 0
        self._n_times_random = 0
        self._init_planning()
        self._cooldown_scaler = self._planning_horizon / .2

    def reset(self):
        self.memory.reset()
        self._init_planning()

    def make_step(self, state, reward, is_done, verbosity: int, info=None):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')
        action = self._make_action(state, is_done, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        self.memory.train(Sa(state, action), verbosity)

        if reward > 0:
            self.planner.add_goal(state)
        return action

    def _init_planning(self):
        self._planned_actions = deque(maxlen=self._planning_horizon + 1)
        self._planning_cooldown = 0
        self._goal_state = None
        self.planner.restore_goal_list()

    def _make_action(self, state, is_done, verbosity: int):
        if self._should_plan and not is_done:
            if self._goal_state is not None:
                if self._goal_state == state:
                    self.planner.remove_goal(self._goal_state)
                self._goal_state = None

            from_sa = Sa(state, None)
            planned_actions, self._goal_state = self.planner.plan_actions(from_sa, verbosity)
            self._planned_actions.extend(planned_actions)
            self._raise_cooldown()
            self._n_times_planned += 1

        if not self._planned_actions:
            self._planned_actions.append(np.random.choice(self._n_actions))
            self._decrease_cooldown()
            self._n_times_random += 1

        return self._planned_actions.popleft()

    @property
    def _should_plan(self):
        enabled, planned = self._planning_enabled, self._planned_actions
        on_cooldown = self._planner_on_cooldown
        return enabled and not planned and not on_cooldown

    @property
    def _planner_on_cooldown(self):
        return self._planning_cooldown > 0

    def _raise_cooldown(self):
        if not self._planned_actions and self._use_cooldown:
            self._planning_cooldown = round(np.random.rand() * self._cooldown_scaler)
            self._planning_cooldown += 1

    def _decrease_cooldown(self):
        if self._planning_enabled and self._use_cooldown:
            self._planning_cooldown -= 1

    @property
    def plan_to_random_ratio(self):
        return (self._n_times_planned + 1) / (self._n_times_random + 1)


class AgentRunner:
    agent: Agent
    env: Any
    n_episodes: int
    max_steps: int
    pretrain: int
    verbosity: int
    train_stats: RunStats
    name: str

    def __init__(self, agent, env, n_episodes, max_steps, pretrain, verbosity):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.pretrain = pretrain
        self.verbosity = verbosity
        self.train_stats = RunStats()
        self.name = '???'

    def run(self):
        trace(self.verbosity, 1, '============> RUN HTM AGENT')
        if self.pretrain > 0:
            planning_horizon = self.agent.planner.planning_horizon
            self.agent.set_planning_horizon(0)

        # log_steps = tqdm(total=0, position=1, bar_format='{desc}')
        for ep in trange(self.n_episodes):
            if 0 < self.pretrain == ep:
                self.agent.set_planning_horizon(planning_horizon)

            (steps, reward), elapsed_time = self.run_episode()
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
        trace(self.verbosity, 1, '<============')

    def run_iterable(self, iterable):
        trace(self.verbosity, 1, '============> RUN HTM AGENT')
        planning_horizon = self.agent.planner.planning_horizon

        for global_ep, local_ep in iterable():
            if self.pretrain > 0 == local_ep:
                self.agent.set_planning_horizon(0)
            if 0 < self.pretrain == local_ep:
                self.agent.set_planning_horizon(planning_horizon)

            (steps, reward), elapsed_time = self.run_episode()
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
            yield
        trace(self.verbosity, 1, '<============')

    @timed
    def run_episode(self):
        self.agent.reset()
        state, reward, done = self.env.reset(), 0, False
        action = self.agent.make_step(state, reward, done, self.verbosity)

        step = 0
        total_reward = 0.
        while step < self.max_steps and not done:
            state, reward, done, info = self.env.step(action)
            # if pretrain, than learn only possible transitions
            # if self.agent.planner.planning_horizon == 0 and (reward <= (-100)):
            #     break
            action = self.agent.make_step(state, reward, done, self.verbosity, info)
            step += 1
            total_reward += reward

        return step, total_reward

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')


class TransferLearningExperimentRunner:
    terminal_states: List[int]
    verbosity: int

    def __init__(self, terminal_states: List[int], verbosity: int):
        self.terminal_states = terminal_states
        self.verbosity = verbosity

    def run_experiment(self, agent_runner: AgentRunner):
        trace(self.verbosity, 1, '========================> RUN TRANSFER LEARNING EXPERIMENT')
        env: Mdp = agent_runner.env
        for terminal_state in self.terminal_states:
            env.terminal_state = terminal_state
            agent_runner.run()
        trace(self.verbosity, 1, '<========================')


class TransferLearningExperimentRunner2:
    env_generator: GridworldMapGenerator
    n_episodes_all_fixed: int
    n_initial_states: int
    n_terminal_states: int
    n_environments: int
    verbosity: int

    def __init__(
            self, env_generator: GridworldMapGenerator, n_episodes_all_fixed: int,
            n_initial_states: int, n_terminal_states: int,
            n_environments: int, verbosity: int
    ):
        self.env_generator = env_generator
        self.n_episodes_all_fixed = n_episodes_all_fixed
        self.n_initial_states = n_initial_states
        self.n_terminal_states = n_terminal_states
        self.n_environments = n_environments
        self.verbosity = verbosity

    def run_experiment(self, agent_runner: AgentRunner):
        n_episodes = self.n_environments * self.n_terminal_states * self.n_initial_states
        n_episodes *= self.n_episodes_all_fixed

        zipped_iterator = lambda: zip(trange(n_episodes), self._get_local_iterator(agent_runner))
        for _ in agent_runner.run_iterable(zipped_iterator):
            ...

    def _get_local_iterator(self, agent_runner: AgentRunner):
        env: GridworldMdp
        for env in islice(self.env_generator, self.n_environments):
            agent_runner.env = env
            for _ in range(self.n_terminal_states):
                env.set_random_terminal_state()
                for _ in range(self.n_initial_states):
                    env.set_random_initial_state()
                    for local_episode in range(self.n_episodes_all_fixed):
                        yield local_episode

    def show_environments(self, n):
        for env_map, _ in self.get_environment_maps(n):
            plt.imshow(env_map)
            plt.show(block=True)

    def get_environment_maps(self, n):
        env: GridworldMdp
        maps = []
        for env in islice(self.env_generator, self.n_environments):
            for _ in range(self.n_terminal_states):
                env.set_random_terminal_state()
                # generate just one initial state instead of self.n_initial_states
                env.set_random_initial_state()
                env.reset()
                maps.append(env.get_representation(mode='img'))
                if len(maps) >= n:
                    return maps
        return maps
