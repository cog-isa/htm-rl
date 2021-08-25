import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from htm_rl.agents.agent import Agent
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.env import Env, unwrap
from htm_rl.scenarios.factories import materialize_environment, materialize_agent
from htm_rl.scenarios.standard.scenario import RunStats
from htm_rl.scenarios.utils import ProgressPoint


class Scenario:
    config: dict

    max_episodes: int
    train_ep_before_eval: int
    mode: str
    n_goals_before_dreaming: int
    goals_found: int
    dreaming_test: int
    stop_dreaming_test: Optional[ProgressPoint]
    dreaming_test_started: bool
    force_dreaming_point: ProgressPoint
    test_forward: int
    compare_vs_last_eps: int

    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(self, config: dict):
        self.config = config
        self.max_episodes = config['max_episodes']
        self.train_ep_before_eval = config['train_ep_before_eval']
        self.n_goals_before_dreaming = config['n_goals_before_dreaming']
        self.dreaming_test = config['dreaming_test']
        self.test_forward = config['test_forward']
        self.compare_vs_last_eps = config['compare_vs_last_eps']
        self.stop_dreaming_test = None
        self.agent_checkpoint = None

        self.init()

    def init(self):
        config = self.config
        self.env = materialize_environment(
            config['envs'][config['env']], config['env_seed']
        )
        self.agent = materialize_agent(
            config['agents'][config['agent']], config['agent_seed'], self.env
        )
        self.mode = 'train'
        self.progress = ProgressPoint()

    def run(self):
        train_stats = RunStats()

        # compute non-dreaming version stats
        while self.progress.episode < self.max_episodes:
            self.run_episode_with_mode(train_stats)

        self.init()
        self.goals_found = 0
        self.dreaming_test_started = False
        dreaming_train_stats = RunStats()
        with tqdm(total=self.max_episodes) as pbar:
            while not self.finish_dreaming_test:
                self.run_episode_dream_test_(dreaming_train_stats, train_stats, pbar)

        plt.plot(train_stats.steps.copy())
        # plt.show()
        return train_stats

    def run_episode_dream_test_(self, dreaming_train_stats, train_stats, pbar):
        def should_announce_progress():
            if not self.dreaming_test_started:
                return True
            return self.progress == self.force_dreaming_point

        def should_start_dreaming_test():
            if self.dreaming_test_started:
                return False
            if self.mode != 'train':
                return False
            return self.goals_found >= self.n_goals_before_dreaming

        if self.dreaming_test_started:
            print(self.mode)

        if should_start_dreaming_test():
            self.init_dreaming_test()
        if self.mode == 'train':
            (_, _, goal_found), _ = self.run_episode_dream_test()
            if self.dreaming_test_started:
                print('ended')
            if goal_found:
                self.goals_found += 1
            if self.should_eval:
                self.switch_to_state('eval')
        elif self.mode == 'eval':
            (steps, reward), elapsed_time = self.run_episode()
            train_stats.append_stats(steps, reward, elapsed_time)
            self.switch_to_state('train')

        if self.mode == 'train':
            self.progress.end_episode()
            if should_announce_progress():
                self.update_progress(pbar)
        elif self.mode == 'eval':
            self.progress.end_episode(increase_episode=False)

        # if self.dreaming_test_started:
        #     print(self.mode)
        # if self.mode == 'train':
        #     if should_start_dreaming_test():
        #         self.init_dreaming_test()
        #
        #     (_, _, goal_found), _ = self.run_episode_dream_test()
        #     if self.dreaming_test_started:
        #         print('ended')
        #     self.progress.end_episode()
        #
        #     if goal_found:
        #         self.goals_found += 1
        #     if should_announce_progress():
        #         self.update_progress(pbar)
        #     if self.should_switch_to_eval:
        #         self.switch_to_state('eval')
        # elif self.mode == 'eval':
        #     (steps, reward), elapsed_time = self.run_episode()
        #     self.progress.end_episode(increase_episode=False)
        #     dreaming_train_stats.append_stats(steps, reward, elapsed_time)
        #     self.switch_to_state('train')

        if not self.dreaming_test_started or self.mode != 'train':
            return

        if self.progress.episode > self.force_dreaming_point.episode + self.test_forward:
            print('reset', self.progress, self.force_dreaming_point)
            return
            # compare and reset back agent
            agent, pos = self.agent_checkpoint
            self.agent = pickle.loads(agent)
            # reset env and set agent position

            env: Environment = unwrap(self.env)
            env.entities['agent'].position = pos

    @timed
    def run_episode_dream_test(self):
        total_reward = 0.
        goal_found = False
        if self.should_force_dreaming_now:
            # cannot force at the start of the episode
            self.force_dreaming_point.next_step()

        while True:
            reward, obs, first = self.env.observe()
            episode_ended = self.episode_ended(first)

            if not self.dreaming_test_started and reward > .1:
                goal_found = True

            if episode_ended and self.should_force_dreaming_now:
                self.force_dreaming_point.end_episode()
                self.force_dreaming_point.next_step()

            if self.dreaming_test_started and self.mode == 'train':
                print(self.progress)

            if self.should_force_dreaming_now:
                print('force')
                self.agent.force_dreaming = True
            action = self.agent.act(reward, obs, first)

            if episode_ended:
                break

            self.env.act(action)
            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward, goal_found

    @property
    def should_force_dreaming_now(self):
        if not self.dreaming_test_started:
            return False
        if not self.mode == 'train':
            return False
        return self.progress == self.force_dreaming_point

    @property
    def finish_dreaming_test(self):
        if self.progress.episode >= self.max_episodes:
            return True

        if not self.dreaming_test_started:
            return False

        return self.force_dreaming_point.episode >= self.stop_dreaming_test.episode

    def init_dreaming_test(self):
        self.dreaming_test_started = True
        self.force_dreaming_point = ProgressPoint(self.progress)
        self.stop_dreaming_test = ProgressPoint(self.progress)
        self.stop_dreaming_test.episode += self.dreaming_test

        print('==>', self.progress, self.force_dreaming_point, self.stop_dreaming_test)

        env: Environment = unwrap(self.env)
        pos = env.entities['agent'].position
        self.agent_checkpoint = (
            pickle.dumps(self.agent),
            pos
        )

    def run_episode_with_mode(self, train_stats, pbar=None):
        self.run_episode()

        if self.should_eval:
            self.switch_to_state('eval')
            self.progress.end_episode(increase_episode=False)
            (steps, reward), elapsed_time = self.run_episode()
            train_stats.append_stats(steps, reward, elapsed_time)
            self.switch_to_state('train')

        self.progress.end_episode()
        self.update_progress(pbar)

    @property
    def should_eval(self):
        return (self.progress.episode + 1) % self.train_ep_before_eval == 0

    def switch_to_state(self, new_state):
        if new_state == 'train':
            self.mode = 'train'
            self.agent.train = True
        elif new_state == 'eval':
            self.mode = 'eval'
            self.agent.train = False

    @staticmethod
    def update_progress(pbar=None):
        if pbar is not None:
            pbar.update(1)

    @timed
    def run_episode(self):
        total_reward = 0.
        while True:
            reward, obs, first = self.env.observe()
            action = self.agent.act(reward, obs, first)
            if self.episode_ended(first):
                break

            self.env.act(action)
            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward

    def episode_ended(self, first: bool):
        return first and self.progress.step > 0
