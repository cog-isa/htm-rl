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
        self.mode = 'train'
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
        self.progress = ProgressPoint()

    def run(self):
        train_stats = RunStats()

        # compute non-dreaming version stats
        while self.progress.episode < self.max_episodes:
            self.run_episode_(train_stats, pbar=None)
        #
        # self.init()
        # self.state = 'train'
        # self.goals_found = 0
        # self.dreaming_test_started = False
        # dreaming_train_stats = RunStats()
        # with tqdm(total=self.max_episodes) as pbar:
        #     while self.progress.episode < self.max_episodes or self.finish_dreaming_test:
        #         self.run_episode_(dreaming_train_stats, pbar)

        plt.plot(train_stats.steps.copy())
        #
        # actions = np.array(self.actions.copy())
        # self.actions = []
        # self.init()
        # self.state = 'train'
        # train_stats = RunStats()
        # while self.progress.episode < self.max_episodes:
        #     self.run_episode_(train_stats, pbar=None)
        #
        # n_actions = np.array(self.actions.copy())
        # # print(actions[:10000])
        # # print(n_actions[:10000])
        # print(np.where(actions != n_actions))
        # assert np.all(actions == n_actions)

        # plt.show()
        return train_stats

    def run_episode_dream_test_(self, dreaming_train_stats, train_stats, pbar):
        if self.mode == 'train':
            print('1')
            (_, r, goal_found), _ = self.run_episode_dream_test()
            self.progress.end_episode()

            print(goal_found, r)
            if goal_found:
                self.goals_found += 1
            if self.goals_found >= self.n_goals_before_dreaming:
                self.init_dreaming_test()

            if (
                    not self.dreaming_test_started
                    or self.progress.episode == self.force_dreaming_point.episode
            ):
                self.update_progress(pbar)

            if self.progress.episode % self.train_ep_before_eval == 0:
                self.mode = 'eval'
                self.agent.train = False
        elif self.mode == 'eval':
            print('2')
            (steps, reward), elapsed_time = self.run_episode()
            dreaming_train_stats.append_stats(steps, reward, elapsed_time)
            self.mode = 'train'
            self.agent.train = True

        if not self.dreaming_test_started:
            return

        if self.force_dreaming_point.episode >= self.progress.episode + self.test_forward:
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

        while True:
            reward, obs, first = self.env.observe()
            if not self.dreaming_test_started and reward > .1:
                print('!!!')
                goal_found = True
            if first and self.progress.step > 0:
                if self.force_dreaming:
                    self.force_dreaming_point.end_episode()
                    self.force_dreaming_point.next_step()
                break

            if self.force_dreaming:
                self.agent.force_dreaming = True
            action = self.agent.act(reward, obs, first)
            self.env.act(action)

            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward, goal_found

    @property
    def force_dreaming(self):
        return (
                self.dreaming_test_started
                and self.progress == self.force_dreaming_point
        )

    @property
    def finish_dreaming_test(self):
        return (
                self.dreaming_test_started
                and self.progress.episode >= self.stop_dreaming_test.episode
        )

    def init_dreaming_test(self):
        print('OK')
        self.dreaming_test_started = True
        self.force_dreaming_point = ProgressPoint(self.progress)
        self.stop_dreaming_test = ProgressPoint(self.progress)
        self.stop_dreaming_test.episode += self.dreaming_test

        env: Environment = unwrap(self.env)
        pos = env.entities['agent'].position
        self.agent_checkpoint = (
            pickle.dumps(self.agent),
            pos
        )

    def run_episode_(self, train_stats, pbar):
        if self.mode == 'train':
            _ = self.run_episode()
            self.progress.end_episode()
            self.update_progress(pbar)
            if self.should_switch_to_eval:
                self.switch_to_state('eval')
        elif self.mode == 'eval':
            (steps, reward), elapsed_time = self.run_episode()
            self.progress.end_episode(increase_episode=False)
            train_stats.append_stats(steps, reward, elapsed_time)
            self.switch_to_state('train')

    @property
    def should_switch_to_eval(self):
        return self.progress.episode % self.train_ep_before_eval == 0

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
            if first and self.progress.step > 0:
                break

            self.env.act(action)
            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward
