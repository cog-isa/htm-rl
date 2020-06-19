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

    @classmethod
    def test(cls):
        for cells_per_column in [1, 2, 3, 5, 8]:
            for path in [[0]*i for i in [1, 2, 3, 4, 5]]:
                assert cls.plan_actions(cells_per_column, path) == path

    @classmethod
    def plan_actions(cls, cells_per_column, path):
        set_random_seed(cls.seed)
        env = make_mdp_passage(cls.cell_gonality, path, cls.seed)
        agent, planner = make_agent(env, cells_per_column, 'full', cls.verbose)

        train(agent, env, n_episodes=100, max_steps=20, verbose=False)

        planned_actions = planner.plan_actions(Sar(env.reset(), None, 0), cls.verbose)
        return planned_actions


def debug_test():
    Test_Planner_Given_2gon_PassageMdp.verbose = True
    Test_Planner_Given_2gon_PassageMdp.plan_actions(2, [0, 0])


if __name__ == '__main__':
    debug_test()