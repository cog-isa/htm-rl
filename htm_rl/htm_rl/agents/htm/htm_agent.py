from htm_rl.agents.htm.hierarchy import Hierarchy, Block, InputBlock, SpatialMemory
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.agents.htm.basal_ganglia import BasalGanglia
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm.bindings.sdr import SDR
import numpy as np
import random


class BioGwLabAction:
    """
    Muscles adapter to BioGwLabEnvironment.
    """
    def __init__(self, muscles_size, n_actions, patterns, noise_tolerance=0.1, do_nothing_action='random'):
        self.muscles_size = muscles_size
        self.n_actions = n_actions
        self.noise_tolerance = noise_tolerance
        self.do_nothing_action = do_nothing_action

        self.patterns = np.array(patterns)

    def get_action(self, sparse_pattern):
        dense_pattern = np.zeros(self.muscles_size)
        dense_pattern[sparse_pattern] = 1

        pattern_sizes = self.patterns.sum(axis=1)
        overlaps = 1 - np.sum(np.abs(self.patterns - dense_pattern), axis=1) / (pattern_sizes + 1e-15)

        if np.any(overlaps >= (1 - self.noise_tolerance)):
            return np.argmax(overlaps)
        else:
            # do nothing action
            if self.do_nothing_action == 'random':
                return np.random.randint(self.n_actions)
            else:
                return self.do_nothing_action


class HTMAgent:
    def __init__(self, config, hierarchy: Hierarchy):
        self.punish_reward = config['punish_reward']
        self.hierarchy = hierarchy
        self.action = BioGwLabAction(**config['action'])

        self.muscles = Muscles(**config['muscles'])

        self.total_patterns = 2**self.action.muscles_size
        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        # proportionality for random generator
        self.alpha = config['alpha']

        self.sp_output = SDR(self.hierarchy.output_block.basal_columns)
        self.sp_input = SDR(self.hierarchy.output_block.get_in_sizes()[-1])

    def make_action(self, state_pattern):
        self.state_pattern.sparse = state_pattern
        # get action from hierarchy
        self.hierarchy.set_input((state_pattern, self.action_pattern))
        hierarchy_action_pattern = self.hierarchy.output_block.get_output('feedback')
        # train muscles
        self.muscles.set_active_muscles(self.action_pattern)
        self.muscles.set_active_input(self.hierarchy.output_block.get_output('basal'))
        self.muscles.depolarize_muscles()
        self.muscles.learn()
        # get muscles activations
        self.muscles.set_active_input(hierarchy_action_pattern)
        self.muscles.depolarize_muscles()
        action_pattern = self.muscles.get_depolarized_muscles()

        self.action_pattern = action_pattern
        # convert muscles activation pattern to environment action
        action = self.action.get_action(action_pattern)
        return action

    def reinforce(self, reward, pseudo_rewards=None):
        """
        Reinforce BasalGanglia.
        :param reward: float:
        Main reward of the environment.
        :param pseudo_rewards: list or None
        Rewards for all blocks in hierarchy, they may differ from actual reward.
        List should be length of number of blocks in hierarchy.
        :return:
        """
        self.hierarchy.output_block.add_reward(reward)
        self.hierarchy.output_block.reinforce()

        if pseudo_rewards is None:
            self.hierarchy.add_rewards([reward] * len(self.hierarchy.blocks))
        else:
            self.hierarchy.add_rewards(pseudo_rewards)

    def reset(self):
        self.hierarchy.reset()
        self.action_pattern = np.empty(0)
        self.state_pattern.sparse = np.empty(0)

    def generate_patterns(self):
        """
        Generate random muscles activation patterns. Number of patterns proportional to patterns in memory.
        :return: numpy array
        """
        n_patterns_to_gen = np.clip(self.alpha * (self.total_patterns - len(self.hierarchy.output_block.sm)),
                                    a_min=0, a_max=self.total_patterns)
        if n_patterns_to_gen > 0:
            return np.random.choice([0, 1], size=(int(n_patterns_to_gen), self.muscles.muscles_size//2))
        else:
            return np.empty(0)

    def train_patterns(self, n_steps=1, train_muscles=True, train_memory=True):
        patterns = self.action.patterns
        for step in range(n_steps):
            for pattern in patterns:
                # train memory
                self.sp_input.dense = pattern
                learn = self.hierarchy.output_block.learn_sp
                self.hierarchy.output_block.sp.compute(self.sp_input, learn, self.sp_output)
                if train_memory:
                    self.hierarchy.output_block.sm.add(self.sp_output.dense.copy())

                # train muscles
                if train_muscles:
                    self.muscles.set_active_muscles(self.sp_input.sparse)
                    self.muscles.set_active_input(self.sp_output.sparse)
                    self.muscles.depolarize_muscles()
                    self.muscles.learn()

            self.hierarchy.output_block.sm.forget()


class HTMAgentRunner:
    def __init__(self, config):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)

        sp_default = config['spatial_pooler_default']
        tm_default = config['temporal_memory_default']
        bg_default = config['basal_ganglia_default']
        sm_default = config['spatial_memory_default']

        block_configs = config['blocks']
        block_default = config['block_default']

        input_block_configs = config['input_blocks']
        input_block_default = config['input_block_default']

        blocks = list()

        for block_conf in input_block_configs:
            blocks.append(InputBlock(**block_conf, **input_block_default))

        for block_conf in block_configs:
            tm = ApicalBasalFeedbackTM(**block_conf['tm'], **tm_default)

            sm = SpatialMemory(**block_conf['sm'], **sm_default)

            if block_conf['sp'] is not None:
                sp = SpatialPooler(**block_conf['sp'], **sp_default)
            else:
                sp = None

            if block_conf['bg'] is not None:
                if block_conf['bg_sp'] is not None:
                    bg_sp = SpatialPooler(**block_conf['bg_sp'], **sp_default)
                else:
                    bg_sp = None
                bg = BasalGanglia(sp=bg_sp, **block_conf['bg'], **bg_default)
            else:
                bg = None

            blocks.append(Block(tm=tm, sm=sm, sp=sp, bg=bg, **block_conf['block'], **block_default))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])

        self.agent = HTMAgent(config['agent'], hierarchy)
        self.environment = BioGwLabEnvironment(**config['environment'])

    def run_episodes(self, n_episodes, verbosity=0, train_patterns=True):
        history = {'steps': list(), 'reward': list()}

        total_reward = 0
        steps = 0
        episode = 0
        while episode < n_episodes:
            reward, obs, is_first = self.environment.observe()

            if is_first:
                self.agent.reset()

                history['steps'].append(steps)
                history['reward'].append(reward)
                steps = 0
                total_reward = 0

                episode += 1
                if verbosity > 0:
                    print(f'episode: {episode}\r')
            else:
                if (reward > 0) and (verbosity > 0):
                    print('nyam')
                self.agent.reinforce(reward)

                steps += 1
                total_reward += reward

            if train_patterns:
                self.agent.train_patterns()

            action = self.agent.make_action(obs)
            self.environment.act(action)

        return history


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt

    with open('../../experiments/htm_agent/htm_runner_config2_test.yaml', 'rb') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = HTMAgentRunner(config)

    plt.imshow(runner.environment.callmethod('render_rgb'))
    plt.show()

    history = runner.run_episodes(500, verbosity=1)

    plt.plot(np.convolve(np.array(history['steps']), np.ones(10)/10, mode='valid'))
    plt.show()

