from itertools import islice
from typing import Any

import numpy as np
from numpy.random._generator import Generator
from tqdm import trange

from htm_rl.agent.mcts_actor_critic import MctsActorCritic
from htm_rl.agent.mcts_planner import MctsPlanner
from htm_rl.agent.memory import Memory
from htm_rl.agent.train_eval import RunStats, RunResultsProcessor
from htm_rl.common.base_sa import Sa
from htm_rl.common.utils import timed, trace
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnv, BioGwLabEnvDynamics, BioGwLabEnvObservationWrapper
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator
from htm_rl.envs.gridworld_pomdp import GridworldPomdp


class MctsAgent:
    memory: Memory
    planner: MctsPlanner
    _n_actions: int
    _planning_horizon: int

    def __init__(
            self, memory: Memory, planner: MctsPlanner,
            mcts_actor_critic: MctsActorCritic, n_actions
    ):
        self.memory = memory
        self.planner = planner
        self._mcts_actor_critic = mcts_actor_critic
        self._n_actions = n_actions

    @property
    def _planning_enabled(self):
        return self.planner.planning_horizon > 0

    def set_planning_horizon(self, planning_horizon: int):
        self.planner.planning_horizon = planning_horizon

    def reset(self):
        self.memory.tm.reset()
        self._mcts_actor_critic.reset()
        # self._trace_mcts_stats()

    def make_step(self, state, reward, is_done, verbosity: int):
        trace(verbosity, 2, f'\nState: {state}; reward: {reward}')

        # TODO remove max(0)
        encoded_state = self.memory.encoder.encode(Sa(state, None))
        self._mcts_actor_critic.add_step(encoded_state, max(reward, 0))

        action = self._make_action(state, verbosity)
        trace(verbosity, 2, f'\nMake action: {action}')

        self.memory.train(Sa(state, action), verbosity)
        return action

    def _make_action(self, state, verbosity: int):
        if self._planning_enabled:
            current_sa = Sa(state, None)
            options = self.planner.predict_states(current_sa, verbosity)
            action = self._mcts_actor_critic.choose(options)
        else:
            action = np.random.choice(self._n_actions)
            trace(3, 1, 'RANDOM')

        return action

    def _trace_mcts_stats(self):
        ac = self._mcts_actor_critic
        q = np.round(ac.cell_value / ac.cell_visited_count, 2)
        value_bit_buckets = (q[i: i + 8] for i in range(0, q.size, 8))
        q_str = '\n'.join(
            ' '.join(map(str, value_bits)) for value_bits in value_bit_buckets
        )
        trace(2, 1, q_str)
        trace(2, 1, '=' * 20)


class MctsAgentRunner:
    agent: MctsAgent
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
        planning_horizon = self.agent.planner.planning_horizon
        if self.pretrain > 0:
            self.agent.set_planning_horizon(0)

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
        encoder = self.agent.planner.encoder
        n_states = self.env.n_states
        states = range(n_states)
        if isinstance(self.env, GridworldPomdp):
            states = [self.env.state_to_obs(state) for state in states]

        states = [encoder.encode(Sa(state, None)) for state in states]
        V_s = np.array([
            self.agent._mcts_actor_critic.value(state, 0)
            for state in states
        ]).reshape((int(np.sqrt(n_states)), -1))
        V_s = (100*V_s).astype(np.int)
        trace(self.verbosity, 1, V_s)

        V_s = self.agent._mcts_actor_critic.value_options(states).reshape((int(np.sqrt(n_states)), -1))
        V_s = V_s.astype(np.int)
        trace(self.verbosity, 1, V_s)

        trace(self.verbosity, 1, '<============')
        print(self.env.initial_state)

    @timed
    def run_episode(self):
        self.agent.reset()
        state, reward, done = self.env.reset(), 0, False
        action = self.agent.make_step(state, reward, done, self.verbosity)

        step = 0
        total_reward = 0.
        while step < self.max_steps and not done:
            state, reward, done, info = self.env.step(action)
            action = self.agent.make_step(state, reward, done, self.verbosity)
            step += 1
            total_reward += reward

        return step, total_reward

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')


class MctsAgentRunner2:
    agent: MctsAgent
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
        trace(self.verbosity, 1, '============> RUN MCTS AGENT')
        planning_horizon = self.agent.planner.planning_horizon
        if self.pretrain > 0:
            self.agent.set_planning_horizon(0)

        for ep in trange(self.n_episodes):
            if 0 < self.pretrain == ep:
                self.agent.set_planning_horizon(planning_horizon)

            (steps, reward), elapsed_time = self.run_episode()
            self.train_stats.append_stats(steps, reward, elapsed_time)
            trace(self.verbosity, 2, '')
        trace(self.verbosity, 1, '<============')

    def run_iterable(self, iterable):
        trace(self.verbosity, 1, '============> RUN MCTS AGENT')
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

        # encoder = self.agent.planner.encoder
        # n_states = self.env.n_states
        # states = range(n_states)
        # if isinstance(self.env, GridworldPomdp):
        #     states = [self.env.state_to_obs(state) for state in states]
        #
        # states = [encoder.encode(Sa(state, None)) for state in states]
        # V_s = np.array([
        #     self.agent._mcts_actor_critic.value(state, 0)
        #     for state in states
        # ]).reshape((int(np.sqrt(n_states)), -1))
        # V_s = (100*V_s).astype(np.int)
        # trace(self.verbosity, 1, V_s)
        #
        # V_s = self.agent._mcts_actor_critic.value_options(states).reshape((int(np.sqrt(n_states)), -1))
        # V_s = V_s.astype(np.int)
        # trace(self.verbosity, 1, V_s)
        #
        # trace(self.verbosity, 1, '<============')
        # print(self.env.initial_state)

    @timed
    def run_episode(self):
        self.agent.reset()
        state, reward, done = self.env.reset(), 0, self.env.is_terminal()
        action = self.agent.make_step(state, reward, done, self.verbosity)

        step = 0
        total_reward = 0.
        while step < self.max_steps and not done:
            state, reward, done, info = self.env.step(action)
            action = self.agent.make_step(state, reward, done, self.verbosity)
            step += 1
            total_reward += reward

        return step, total_reward

    def store_results(self, run_results_processor: RunResultsProcessor):
        run_results_processor.store_result(self.train_stats, f'{self.name}')


class TransferLearningExperimentRunner3:
    rnd: Generator
    env_generator: BioGwLabEnvGenerator
    n_episodes_all_fixed: int
    n_initial_states: int
    n_environments: int
    verbosity: int

    def __init__(
            self, env_generator: BioGwLabEnvGenerator, n_episodes_all_fixed: int,
            n_initial_states: int, n_terminal_states: int,
            n_environments: int, verbosity: int
    ):
        self.env_generator = env_generator
        self.n_episodes_all_fixed = n_episodes_all_fixed
        self.n_initial_states = n_initial_states
        self.n_terminal_states = n_terminal_states
        self.n_environments = n_environments
        self.rnd = np.random.default_rng(seed=env_generator.seed)
        self.verbosity = verbosity

    def run_experiment(self, agent_runner: MctsAgentRunner2):
        n_episodes = self.n_environments * self.n_terminal_states * self.n_initial_states
        n_episodes *= self.n_episodes_all_fixed

        zipped_iterator = lambda: zip(trange(n_episodes), self._get_local_iterator(agent_runner))
        for _ in agent_runner.run_iterable(zipped_iterator):
            ...

    def _get_local_iterator(self, agent_runner: MctsAgentRunner2):
        env: BioGwLabEnv
        dynamics = BioGwLabEnvDynamics()
        view_rect = (-2, 0), (2, 2)
        scent_rect = (-3, -2), (3, 4)

        for i in range(self.n_environments):
            seed = self.rnd.integers(100000)
            self.env_generator.seed = seed
            state = self.env_generator.generate()
            env = BioGwLabEnvObservationWrapper(state, dynamics, view_rect, scent_rect)
            agent_runner.env = env
            for _ in range(self.n_initial_states):
                env.generate_initial_position()
                for local_episode in range(self.n_episodes_all_fixed):
                    yield local_episode
