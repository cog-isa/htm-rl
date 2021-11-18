import numpy as np

from htm_rl.agents.hima.adapter import BioGwLabAction
from htm_rl.agents.hima.hierarchy import Hierarchy
from htm_rl.modules.htm.muscles import Muscles
from htm_rl.modules.dreaming.dreamer import Dreamer
from htm_rl.modules.empowerment import Empowerment

from htm.bindings.sdr import SDR


class HIMA:
    def __init__(self, config, hierarchy: Hierarchy):
        self.use_intrinsic_reward = config['use_intrinsic_reward']
        self.use_dreaming = config['use_dreaming']
        self.punish_intrinsic_reward = config['punish_intrinsic_reward']
        self.hierarchy = hierarchy

        self.action = BioGwLabAction(**config['action'])

        self.muscles = Muscles(**config['muscles'])

        if self.use_intrinsic_reward:
            self.empowerment_horizon = config['empowerment'].pop('horizon')
            self.empowerment = Empowerment(**config['empowerment'])
        else:
            self.empowerment_horizon = 0
            self.empowerment = None

        if self.use_dreaming:
            self.dreamer = Dreamer(
                n_actions=self.action.n_actions,
                agent=self,
                state_encoder=self.hierarchy.visual_block.sp,
                **config['dreaming']
            )
        else:
            self.dreamer = None

        self.total_patterns = 2 ** self.action.muscles_size
        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        # proportionality for random generator
        self.alpha = config['alpha']

        self.sp_output = SDR(self.hierarchy.output_block.basal_columns)
        self.sp_input = SDR(self.hierarchy.output_block.get_in_sizes()[-1])
        self.previous_obs = np.empty(0)
        self.backup = None
        self.real_pos = None

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

        # train empowerment tm
        # if self.use_intrinsic_reward:
        #     current_obs = self.hierarchy.visual_block.get_output('basal')
        #     if (self.previous_obs.size > 0) and (current_obs.size > 0) and self.empowerment.filename is None:
        #         self.empowerment.learn(self.previous_obs, current_obs)
        #     self.previous_obs = current_obs
        return action

    def get_intrinsic_reward(self):
        if self.empowerment.filename is None:
            state = self.hierarchy.visual_block.get_output('basal')
            reward = self.empowerment.eval_state(state, self.empowerment_horizon,
                                             use_memory=True)[0]
        else:

            reward = self.empowerment.eval_from_file(self.real_pos)
        return reward

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
