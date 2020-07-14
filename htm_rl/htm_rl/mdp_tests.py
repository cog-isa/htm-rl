from htm_rl.agent.train_eval import train, train_trajectories
from htm_rl.common.base_sar import Sar
from htm_rl.config import set_random_seed, make_mdp_passage, make_agent

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

    def test(self):
        for cells_per_column in [1, 2, 6]:
            for path in [[0]*i for i in [1, 2, 3, 4]]:
                assert self.plan_actions(cells_per_column, path) == path

    # TODO: make plan_actions to accept n_episodes and max_steps and make the test commented below
    # def test_large(self):
    #     path = [0]*5
    #     for cells_per_column in [1, 2, 6]:
    #         assert self.plan_actions(cells_per_column, path) == path
    #         train(agent, env, n_episodes=80 * len(path), max_steps=40, verbose=False)

    def plan_actions(self, cells_per_column, path):
        set_random_seed(self.seed)
        env = make_mdp_passage(self.cell_gonality, path, self.seed)
        agent = make_agent(env, cells_per_column, 'full', self.verbose)

        train(agent, env, n_episodes=40, max_steps=20, verbose=False)

        planned_actions = agent.plan_actions(Sar(env.reset(), None, 0), self.verbose)
        return planned_actions


def debug_test():
    Test_Planner_Given_2gon_PassageMdp.verbose = True
    Test_Planner_Given_2gon_PassageMdp.cell_gonality = 4
    Test_Planner_Given_2gon_PassageMdp().plan_actions(1, [0, 1, 0])


if __name__ == '__main__':
    debug_test()