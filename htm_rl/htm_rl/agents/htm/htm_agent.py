from htm_rl.agents.htm.hierarchy import Hierarchy, SpatialMemory, Block, InputBlock
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.agents.htm.basal_ganglia import BasalGanglia, softmax
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm.bindings.sdr import SDR
import numpy as np


class BioGwLabAction:
    """
    Muscles adapter to BioGwLabEnvironment.
    """
    def __init__(self, muscles_size, n_actions, noise_tolerance=0.1, patterns=None):
        self.muscles_size = muscles_size
        self.n_actions = n_actions
        self.noise_tolerance = noise_tolerance

        if patterns is None:
            # generate random pattern for every action
            self.patterns = np.random.choice([0, 1], size=(n_actions, self.muscles_size))
        else:
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
            return self.n_actions - 1


class HTMAgent:
    def __init__(self, config, hierarchy: Hierarchy):
        self.punish_reward = config['punish_reward']
        self.hierarchy = hierarchy
        self.action = BioGwLabAction(**config['action'])

        self.memory = SpatialMemory(**config['sm'])

        if config['bg_sp'] is not None:
            bg_sp = SpatialPooler(**config['bg_sp'])
        else:
            bg_sp = None
        self.bg = BasalGanglia(sp=bg_sp, **config['bg'])

        self.muscles = Muscles(**config['muscles'])

        self.total_patterns = 2**self.muscles.muscles_size
        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        self.hierarchy_made_decision = False
        # proportionality for random generator
        self.alpha = config['alpha']

    def make_action(self, state_pattern):
        self.state_pattern.sparse = state_pattern
        # add new patterns to memory
        new_patterns = self.generate_patterns()
        for pattern in new_patterns:
            self.memory.add(pattern)

        # get action from hierarchy
        self.hierarchy.set_input((state_pattern, self.action_pattern))
        hierarchy_action_pattern, hierarchy_value = self.hierarchy.output_block.get_output('feedback', return_value=True)
        # train muscles
        self.muscles.set_active_muscles(self.action_pattern)
        self.muscles.set_active_input(self.hierarchy.output_block.get_output('basal'))
        self.muscles.depolarize_muscles()
        self.muscles.learn()
        # get action from lowest bg
        options = self.memory.get_sparse_patterns()
        bg_action_pattern, bg_value, option_values = self.bg.choose(options, condition=self.state_pattern,
                                                                    return_option_value=True, return_values=True)
        # train memory
        self.memory.reinforce(option_values)
        # decide which pattern to use
        if hierarchy_value is None:
            action_pattern = bg_action_pattern
            self.hierarchy_made_decision = False
        else:
            prob = softmax(np.array([hierarchy_value, bg_value]))  # ???
            gamma = np.random.random()
            if gamma < prob[0]:
                self.muscles.set_active_input(hierarchy_action_pattern)
                self.muscles.depolarize_muscles()
                action_pattern = self.muscles.get_depolarized_muscles()
                self.hierarchy_made_decision = True
                self.hierarchy.rejected = False
            else:
                action_pattern = bg_action_pattern
                self.hierarchy_made_decision = False
                self.hierarchy.rejected = True

        self.action_pattern = action_pattern
        # convert muscles activation pattern to environment action
        action = self.action.get_action(action_pattern)
        return action

    def reinforce(self, reward, pseudo_rewards=None, punish_for_muscles_activation = False):
        """
        Reinforce BasalGanglia.
        :param reward: float:
        Main reward of the environment.
        :param pseudo_rewards: list or None
        Rewards for all blocks in hierarchy, they may differ from actual reward.
        List should be length of number of blocks in hierarchy.
        :return:
        """
        if punish_for_muscles_activation:
            reward += (len(self.action_pattern) * (self.punish_reward/self.muscles.muscles_size))

        if self.hierarchy_made_decision:
            self.hierarchy.output_block.add_reward(reward)
            self.hierarchy.output_block.reinforce()

            self.bg.force_dopamine(self.punish_reward)  # ??
        else:
            self.hierarchy.output_block.add_reward(self.punish_reward)  # ??
            self.hierarchy.output_block.reinforce()

            self.bg.force_dopamine(reward)

        if pseudo_rewards is None:
            self.hierarchy.add_rewards([reward] * len(self.hierarchy.blocks))
        else:
            self.hierarchy.add_rewards(pseudo_rewards)

    def reset(self):
        self.hierarchy.reset()
        self.action_pattern = np.empty(0)
        self.state_pattern.sparse = np.empty(0)
        self.hierarchy_made_decision = False

    def generate_patterns(self):
        """
        Generate random muscles activation patterns. Number of patterns proportional to patterns in memory.
        :return: numpy array
        """
        n_patterns_to_gen = np.clip(self.alpha * (self.total_patterns - len(self.memory)),
                                    a_min=0, a_max=self.total_patterns)
        if n_patterns_to_gen > 0:
            return np.random.choice([0, 1], size=(int(n_patterns_to_gen), self.muscles.muscles_size))
        else:
            return np.empty(0)


class HTMAgentRunner:
    def __init__(self, config):
        sp_default = config['spatial_pooler_default']
        tm_default = config['temporal_memory_default']
        bg_default = config['basal_ganglia_default']

        block_configs = config['blocks']
        block_default = config['block_default']

        input_block_configs = config['input_blocks']
        input_block_default = config['input_block_default']

        blocks = list()

        for block_conf in input_block_configs:
            blocks.append(InputBlock(**block_conf, **input_block_default))

        for block_conf in block_configs:
            tm = ApicalBasalFeedbackTM(**block_conf['tm'], **tm_default)

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

            blocks.append(Block(tm=tm, sp=sp, bg=bg, **block_conf['block'], **block_default))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])

        self.agent = HTMAgent(config['agent'], hierarchy)
        self.environment = BioGwLabEnvironment(**config['environment'])

    def run_episode(self):
        pass


if __name__ == '__main__':
    import yaml
    with open('../../experiments/htm_agent/htm_runner_config_test.yaml', 'rb') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = HTMAgentRunner(config)

    reward, state, _ = runner.environment.observe()

    for i in range(200):
        # print(runner.environment.callmethod('render_rgb'))

        action = runner.agent.make_action(state)
        runner.environment.act(action)

        reward, state, _ = runner.environment.observe()
        runner.agent.reinforce(reward)
        # print(f'action:{action}, reward: {reward}')
    print(runner.agent.memory.get_sparse_patterns())
    print(runner.agent.hierarchy.output_block.patterns.get_sparse_patterns())
    print(runner.agent.hierarchy.blocks[5].patterns.get_sparse_patterns())


