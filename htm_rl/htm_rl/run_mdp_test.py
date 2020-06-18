from htm_rl.agent.train_eval import train
from htm_rl.common.base_sar import Sar
from htm_rl.config import set_random_seed, make_mdp_one_way_snake, make_agent

"""
Here you can test agent+planner on a simple handcrafted MDP envs.
"""


def run_test():
    seed = 1337
    verbose = True

    set_random_seed(seed)

    env = make_mdp_one_way_snake(start_direction=2, n_cells=3, seed=seed)
    agent, planner = make_agent(env, verbose)

    train(agent, env, n_episodes=30, max_steps=50, verbose=False)
    planner.plan_actions(Sar(env.reset(), None, 0), True)


if __name__ == '__main__':
    run_test()