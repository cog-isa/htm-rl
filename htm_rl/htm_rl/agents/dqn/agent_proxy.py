import numpy as np
from htm.bindings.sdr import SDR

from htm_rl.agents.htm.hierarchy import Hierarchy
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.modules.dreaming.dreamer import Dreamer
from htm_rl.modules.empowerment import Empowerment


class DqnAgentProxy:
    def __init__(self, config, env):

        self.total_patterns = 2 ** self.action.muscles_size
        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        # proportionality for random generator
        self.alpha = config['alpha']

        self.sp_input = SDR(self.hierarchy.output_block.get_in_sizes()[-1])
        self.previous_obs = np.empty(0)
        self.backup = None
        self.real_pos = None

    def make_action(self, state_pattern):
        self.state_pattern.sparse = state_pattern
        action = -1
        return action

    def reinforce(self, reward: float):
        """
        Reinforce BasalGanglia.
        :param reward: float:
        Main reward of the environment.
        Rewards for all blocks in hierarchy, they may differ from actual reward.
        List should be length of number of blocks in hierarchy.
        :return:
        """
        if self.use_intrinsic_reward:
            reward_int = self.get_intrinsic_reward() + self.punish_intrinsic_reward
        else:
            reward_int = 0

        self.hierarchy.add_rewards([reward] * len(self.hierarchy.blocks),
                                   [reward_int] * len(self.hierarchy.blocks))

    def reset(self):
        self.hierarchy.reset()
        self.action_pattern = np.empty(0)
        self.state_pattern.sparse = np.empty(0)
        self.previous_obs = np.empty(0)

    def generate_patterns(self):
        """
        Generate random muscles activation patterns. Number of patterns proportional to patterns in memory.
        :return: numpy array
        """
        n_patterns_to_gen = np.clip(self.alpha * (self.total_patterns - len(self.hierarchy.output_block.sm)),
                                    a_min=0, a_max=self.total_patterns)
        if n_patterns_to_gen > 0:
            return np.random.choice([0, 1], size=(int(n_patterns_to_gen), self.muscles.muscles_size // 2))
        else:
            return np.empty(0)

    def train_patterns(self, n_steps=1, train_muscles=True, train_memory=True):
        patterns = self.action.patterns
        for step in range(n_steps):
            for pattern in patterns:
                # train memory
                if self.hierarchy.output_block.sp is not None:
                    self.sp_input.dense = pattern
                    learn = self.hierarchy.output_block.learn_sp
                    self.hierarchy.output_block.sp.compute(self.sp_input, learn, self.sp_output)
                    if train_memory:
                        self.hierarchy.output_block.sm.add(self.sp_output.dense.copy())
                elif train_memory:
                    self.hierarchy.output_block.sm.add(pattern)
                    self.sp_input.dense = pattern
                    self.sp_output.dense = pattern

                # train muscles
                if train_muscles:
                    self.muscles.set_active_muscles(self.sp_input.sparse)
                    self.muscles.set_active_input(self.sp_output.sparse)
                    self.muscles.depolarize_muscles()
                    self.muscles.learn()

            self.hierarchy.output_block.sm.forget()

    def backup_agent(self):
        pass

    def restore_agent(self):
        pass