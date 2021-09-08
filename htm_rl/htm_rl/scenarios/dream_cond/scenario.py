import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm

from htm_rl.agents.agent import Agent
from htm_rl.agents.dreamer.debug.dream_cond_debugger import DreamingConditionsDebugger
from htm_rl.common.utils import timed
from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.env import Env, unwrap
from htm_rl.scenarios.factories import materialize_environment, materialize_agent
from htm_rl.scenarios.standard.run_stats import RunStats
from htm_rl.scenarios.utils import ProgressPoint


class Scenario:
    config: dict

    max_episodes: int
    train_ep_before_eval: int
    n_goals_before_dreaming: int
    goals_found: int
    dreaming_test: int
    stop_dreaming_test: Optional[ProgressPoint]
    force_dreaming_point: ProgressPoint
    test_forward: int
    compare_vs_last_eps: int

    debug: dict
    debug_enabled: bool

    mode: str
    env: Env
    agent: Agent
    progress: ProgressPoint

    def __init__(
            self, config: dict, max_episodes: int, train_ep_before_eval: int,
            n_goals_before_dreaming: int, dreaming_test: int, test_forward: int,
            compare_vs_last_eps: int, debug: dict,
            **_
    ):
        self.config = config
        self.max_episodes = max_episodes
        self.train_ep_before_eval = train_ep_before_eval
        self.n_goals_before_dreaming = n_goals_before_dreaming
        self.dreaming_test = dreaming_test
        self.test_forward = test_forward
        self.compare_vs_last_eps = compare_vs_last_eps

        self.debug = debug
        self.debug_enabled = debug['enabled']

        self.stop_dreaming_test = None
        self.agent_checkpoint = None
        self.init_run()
        self.test_results_map = dict()

    def init_run(self, with_debug=False):
        config = self.config
        self.env = materialize_environment(
            config['envs'][config['env']], config['env_seed']
        )
        self.agent = materialize_agent(
            config['agents'][config['agent']], config['agent_seed'], self.env
        )
        self.mode = 'train'
        self.progress = ProgressPoint()

    def run(self) -> tuple[RunStats, RunStats]:
        no_dreaming_train_stats = RunStats()
        no_dreaming_eval_stats = RunStats()

        if self.debug_enabled:
            DreamingConditionsDebugger(self, self.debug['images'])

        # compute non-dreaming version stats
        while self.progress.episode < self.max_episodes:
            self.run_episode_with_mode(
                no_dreaming_train_stats, no_dreaming_eval_stats
            )

        # return no_dreaming_train_stats, no_dreaming_eval_stats

        # start testing dreaming
        self.init_run()
        self.goals_found = 0
        dreaming_train_stats = RunStats()
        dreaming_eval_stats = RunStats()
        with tqdm(total=self.max_episodes) as pbar:
            while not self.should_proceed_to_dreaming_test:
                self.run_episode_with_mode_count_goals(
                    dreaming_train_stats, dreaming_eval_stats
                )
                pbar.update(1)

            self.init_dreaming_test()
            while not self.should_finish_dreaming_test:
                self.run_episode_with_mode_dreaming_test(
                    dreaming_train_stats, dreaming_eval_stats,
                    no_dreaming_train_stats, no_dreaming_eval_stats,
                )
                if self.should_force_dreaming_now:
                    # start of the episode ==> force dreaming point
                    # was just moved to the next episode
                    pbar.update(1)

        # plt.plot(train_stats.steps.copy())
        # plt.show()

        changes = [
            # (pp, pos, int(ds))
            int(ds)
            for pp, (pos, ds) in self.test_results_map.items()
            if ds != 0.
        ]

        print(f'Tested {len(self.test_results_map)} dreaming positions. Changes:')
        print(changes)
        return no_dreaming_train_stats, no_dreaming_eval_stats

    def run_episode_with_mode_dreaming_test(
            self, train_stats, eval_stats,
            train_stats_no_dreaming, eval_stats_no_dreaming,
    ):
        (steps, reward), elapsed_time = self.run_episode_with_dream_forcing()
        train_stats.append_stats(steps, reward, elapsed_time)
        # print('ended')

        if self.should_eval:
            self.progress.end_episode(increase_episode=False)
            self.switch_to_state('eval')
            (steps, reward), elapsed_time = self.run_episode()
            eval_stats.append_stats(steps, reward, elapsed_time)
            self.progress.end_episode()
            self.switch_to_state('train')
        else:
            self.progress.end_episode()

        if self.progress.episode > self.force_dreaming_point.episode + self.test_forward:
            # print('reset ==>', self.progress, self.force_dreaming_point)
            self.save_results(eval_stats, eval_stats_no_dreaming)
            self.cut_stats_to_checkpoint(train_stats, eval_stats)
            self.restore_checkpoint()
            self.proceed_to_the_next_test(train_stats_no_dreaming)
            # print('reset <==', self.progress, self.force_dreaming_point)

    @timed
    def run_episode_with_dream_forcing(self):
        total_reward = 0.
        if self.progress.step == 0 and self.should_force_dreaming_now:
            # cannot force at the start of the episode
            self.force_dreaming_point.next_step()

        while True:
            # print(self.progress)
            reward, obs, first = self.env.observe()
            episode_ended = self.episode_ended(first)

            if self.should_force_dreaming_now:
                # print('save checkpoint')
                self.save_checkpoint()
                # print('force')
                self.agent.force_dreaming = True
            action = self.agent.act(reward, obs, first)

            if episode_ended:
                break

            self.env.act(action)
            self.progress.next_step()
            total_reward += reward

        return self.progress.step, total_reward

    @property
    def should_force_dreaming_now(self):
        return self.progress == self.force_dreaming_point

    @property
    def should_finish_dreaming_test(self):
        if self.progress.episode >= self.max_episodes:
            return True
        return self.force_dreaming_point.episode >= self.stop_dreaming_test.episode

    def restore_checkpoint(self):
        # compare and reset back agent
        agent, pos, episode_step, step_reward = self.agent_checkpoint
        self.agent = pickle.loads(agent)

        # reset env and set agent position
        env: Environment = unwrap(self.env)
        # noinspection PyUnresolvedReferences
        env.entities['agent'].position = pos
        env.episode_step = episode_step
        env.step_reward = step_reward

        # HACK: append agent position to the result
        pp = self.force_dreaming_point
        self.test_results_map[pp] = (pos, self.test_results_map[pp])

        # set current progress and new dreaming point
        self.progress = ProgressPoint(self.force_dreaming_point)

    def proceed_to_the_next_test(self, train_stats_no_dreaming: RunStats):
        self.force_dreaming_point.next_step()
        # move force dreaming to the next episode if needed
        episode_end_step = train_stats_no_dreaming.steps[self.progress.episode]
        if self.force_dreaming_point.step >= episode_end_step - 1:
            # at `end - 1` step achieves goal, no need for dreaming
            self.force_dreaming_point.end_episode()

    def cut_stats_to_checkpoint(self, train_stats: RunStats, eval_stats: RunStats):
        # to reuse existing stats, we should cut them back to the
        # starting point - force_dreaming
        n = self.test_forward + 1
        for stats in [train_stats, eval_stats]:
            for lst in [stats.rewards, stats.steps, stats.times]:
                del lst[-n:]

        # print(len(train_stats.steps), self.force_dreaming_point.episode)
        # assert len(train_stats.steps) == self.force_dreaming_point.episode

    def save_results(self, dreaming_stats: RunStats, no_dreaming_stats: RunStats):
        # DO NOT COMPARE WITH REWARDS, BECAUSE THERE'S A BUG WITH IT:
        # I DON'T RESTORE CUMULATIVE EPISODE REWARD FOR THE DREAMING EPISODE
        # === but we restore episode `step`, and it's enough to compare :)
        avg_dreaming_res = np.mean(dreaming_stats.steps[-self.compare_vs_last_eps:])
        avg_no_dreaming_res = np.mean(no_dreaming_stats.steps[-self.compare_vs_last_eps:])
        # print(avg_no_dreaming_res, avg_dreaming_res)
        diff = avg_dreaming_res - avg_no_dreaming_res
        self.test_results_map[self.force_dreaming_point] = diff

    def save_checkpoint(self):
        env: Environment = unwrap(self.env)
        # noinspection PyUnresolvedReferences
        pos = env.entities['agent'].position
        self.agent_checkpoint = (
            pickle.dumps(self.agent),
            pos,
            env.episode_step,
            env.step_reward
        )

    def init_dreaming_test(self):
        self.force_dreaming_point = ProgressPoint(self.progress)
        self.stop_dreaming_test = ProgressPoint(self.progress)
        self.stop_dreaming_test.episode += self.dreaming_test
        # print('==>', self.progress, self.force_dreaming_point, self.stop_dreaming_test)

    @property
    def should_proceed_to_dreaming_test(self):
        if self.progress.episode >= self.max_episodes:
            # end scenario
            return True
        return self.goals_found >= self.n_goals_before_dreaming

    def run_episode_with_mode_count_goals(self, train_stats, eval_stats):
        (steps, reward), elapsed_time = self.run_episode()
        train_stats.append_stats(steps, reward, elapsed_time)
        if reward > .02:
            self.goals_found += 1

        if self.should_eval:
            self.progress.end_episode(increase_episode=False)
            self.switch_to_state('eval')
            (steps, reward), elapsed_time = self.run_episode()
            eval_stats.append_stats(steps, reward, elapsed_time)
            self.progress.end_episode()
            self.switch_to_state('train')
        else:
            self.progress.end_episode()

    def run_episode_with_mode(self, train_stats, eval_stats):
        (steps, reward), elapsed_time = self.run_episode()
        train_stats.append_stats(steps, reward, elapsed_time)

        if self.should_eval:
            self.progress.end_episode(increase_episode=False)
            self.switch_to_state('eval')
            (steps, reward), elapsed_time = self.run_episode()
            eval_stats.append_stats(steps, reward, elapsed_time)
            self.progress.end_episode()
            self.switch_to_state('train')
        else:
            self.progress.end_episode()

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
