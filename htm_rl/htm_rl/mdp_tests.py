from timeit import default_timer as timer

import numpy

from htm_rl.agent.train_eval import train, plot_train_results
from htm_rl.config import set_random_seed, make_mdp_passage, make_agent, make_mdp_multi_way_v0

"""
Here you can test agent+planner on a simple handcrafted MDP envs.
"""


class TestConfig:
    random_seed: int = 1337
    verbosity: int = 0


class Test_Planner_Given_2gon_PassageMdp:
    seed: int = 1337
    verbose: bool = False
    cell_gonality: int = 2

    # TODO: make plan_actions to accept n_episodes and max_steps and make the test commented below
    # def test(self):
    #     for cells_per_column in [1, 2, 6]:
    #         for path in [[0]*i for i in [1, 2, 3, 4]]:
    #             assert self.plan_actions(cells_per_column, path) == path
    # def test_large(self):
    #     path = [0]*5
    #     for cells_per_column in [1, 2, 6]:
    #         assert self.plan_actions(cells_per_column, path) == path
    #         train(agent, env, n_episodes=80 * len(path), max_steps=40, verbose=False)

    # def plan_actions(self, cells_per_column, path):
    #     set_random_seed(self.seed)
    #     env = make_mdp_passage(self.cell_gonality, path, self.seed)
    #     agent = make_agent(env, cells_per_column, 'full', self.verbose)
    #
    #     train(agent, env, n_episodes=40, max_steps=20, verbose=False)
    #
    #     planned_actions = agent.plan_actions(Sar(env.reset(), None, 0), self.verbose)
    #     return planned_actions

    def plan_actions(self, cells_per_column, path):
        set_random_seed(self.seed)
        start = timer()

        n_episodes, max_steps, planning_horizon, clockwise, cooldown = 20, 20, 1, False, False

        env = make_mdp_passage(self.cell_gonality, path, self.seed, clockwise_action=clockwise)
        agent = make_agent(
            env, cells_per_column, planning_horizon, cooldown, self.seed, 'short', self.verbose
        )
        episodes_len, rewards = train(
            agent, env, n_episodes=n_episodes, max_steps=max_steps, verbose=False
        )
        elapsed = timer() - start
        episode_t_mean = elapsed / n_episodes
        # train(agent, env, n_episodes=1, max_steps=max_steps, verbose=True)

        rewards /= episodes_len
        plan_to_random = agent.plan_to_random_ratio
        print(f'T: {elapsed: .4f}, EpT: {episode_t_mean: .4f}, P2R: {plan_to_random: .4f}')
        plot_title = f'Episodes: {n_episodes}, MaxSteps: {max_steps}, PlanningHorizon: {planning_horizon}, ' \
                     f'ClockwiseAction: {clockwise}, UseCooldown: {cooldown}'
        episodes_title = f'Episodes length, mean: {episodes_len.mean(): .3f}'
        rewards_title = f'Discounted reward, mean: {rewards.mean(): .3f}'
        print(plot_title)
        print(episodes_title)
        print(rewards_title)
        # plot_train_results(episodes_len, rewards, plot_title, episodes_title, rewards_title)


class Test_Planner_Given_4gon_PassageMdp:
    seed: int = 1337
    verbose: bool = False
    cell_gonality: int = 4

    def plan_actions(self, cells_per_column, path):
        set_random_seed(self.seed)
        start = timer()

        n_episodes, max_steps, planning_horizon, clockwise, cooldown = 50, 80, 6, True, True

        env = make_mdp_passage(self.cell_gonality, path, self.seed, clockwise_action=clockwise)
        agent = make_agent(
            env, cells_per_column, planning_horizon, cooldown, self.seed, 'short', self.verbose
        )
        episodes_len, rewards = train(
            agent, env, n_episodes=n_episodes, max_steps=max_steps, verbose=False
        )
        elapsed = timer() - start
        episode_t_mean = elapsed / n_episodes
        # train(agent, env, n_episodes=1, max_steps=max_steps, verbose=True)

        rewards /= episodes_len
        plan_to_random = agent.plan_to_random_ratio
        print(f'T: {elapsed: .4f}, EpT: {episode_t_mean: .4f}, P2R: {plan_to_random: .4f}')
        plot_title = f'Episodes: {n_episodes}, MaxSteps: {max_steps}, PlanningHorizon: {planning_horizon}, ' \
                     f'ClockwiseAction: {clockwise}, UseCooldown: {cooldown}'
        episodes_title = f'Episodes length, mean: {episodes_len.mean(): .3f}'
        rewards_title = f'Discounted reward, mean: {rewards.mean(): .3f}'
        print(plot_title)
        print(episodes_title)
        print(rewards_title)
        # plot_train_results(episodes_len, rewards, plot_title, episodes_title, rewards_title)


class Test_Planner_Given_Labirinth_v0:
    seed: int = 1337
    verbose: bool = False

    def plan_actions(self, cells_per_column):
        set_random_seed(self.seed)
        start = timer()

        n_episodes, max_steps, planning_horizon, clockwise, cooldown = 50, 100, 4, False, False
        env = make_mdp_multi_way_v0(0, self.seed, clockwise_action=clockwise)
        agent = make_agent(
            env, cells_per_column, planning_horizon, cooldown, self.seed, 'short', self.verbose
        )
        episodes_len, rewards = train(
            agent, env, n_episodes=n_episodes, max_steps=max_steps, verbose=False
        )
        elapsed = timer() - start
        episode_t_mean = elapsed / n_episodes
        # train(agent, env, n_episodes=1, max_steps=max_steps, verbose=True)

        rewards /= episodes_len
        plan_to_random = agent.plan_to_random_ratio
        print(f'T: {elapsed: .4f}, EpT: {episode_t_mean: .4f}, P2R: {plan_to_random: .4f}')
        plot_title = f'Episodes: {n_episodes}, MaxSteps: {max_steps}, PlanningHorizon: {planning_horizon}, ' \
                     f'ClockwiseAction: {clockwise}, UseCooldown: {cooldown}'
        episodes_title = f'Episodes length, mean: {episodes_len.mean(): .3f}'
        rewards_title = f'Discounted reward, mean: {rewards.mean(): .3f}'
        print(plot_title)
        print(episodes_title)
        print(rewards_title)
        plot_train_results(episodes_len, rewards, plot_title, episodes_title, rewards_title)


def debug_test():
    # Test_Planner_Given_4gon_PassageMdp.verbose = True
    # Test_Planner_Given_4gon_PassageMdp().plan_actions(1, [0, 1, 0])
    Test_Planner_Given_Labirinth_v0().plan_actions(1)


if __name__ == '__main__':
    debug_test()