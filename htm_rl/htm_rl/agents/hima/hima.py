import numpy as np

from htm_rl.agents.hima.elementary_actions import ElementaryActions
from htm_rl.agents.hima.hierarchy import Hierarchy
from htm_rl.modules.dreaming.dreamer import Dreamer
from htm_rl.modules.empowerment import Empowerment

from htm.bindings.sdr import SDR


class HIMA:
    def __init__(self, config, hierarchy: Hierarchy):
        self.use_intrinsic_reward = config['use_intrinsic_reward']
        self.use_dreaming = config['use_dreaming']
        self.punish_intrinsic_reward = config['punish_intrinsic_reward']
        self.hierarchy = hierarchy
        self.elementary_actions = ElementaryActions(**config['elementary_actions'])

        # initialize and pattern memory for output block
        for pattern in self.elementary_actions.patterns:
            self.hierarchy.output_block.sm.add(pattern)

        self.n_actions_to_accumulate = config['n_actions_to_accumulate']
        self.accumulated_action = np.empty(0)

        if self.use_intrinsic_reward:
            self.empowerment_horizon = config['empowerment'].pop('horizon')
            self.empowerment = Empowerment(**config['empowerment'])
        else:
            self.empowerment_horizon = 0
            self.empowerment = None

        if self.use_dreaming:
            self.dreamer = Dreamer(
                n_actions=self.elementary_actions.n_actions,
                agent=self,
                state_encoder=self.hierarchy.visual_block.sp,
                **config['dreaming']
            )
        else:
            self.dreamer = None

        # there is no first action
        self.action_pattern = np.empty(0)
        self.state_pattern = SDR(config['state_size'])

        self.sp_output = SDR(self.hierarchy.output_block.basal_columns)
        self.sp_input = SDR(self.hierarchy.output_block.get_in_sizes()[-1])
        self.previous_obs = np.empty(0)
        self.backup = None
        self.real_pos = None

    def make_action(self, state_pattern):
        self.accumulated_action = np.empty(0)
        self.state_pattern.sparse = state_pattern
        shift = 0
        for n in range(self.n_actions_to_accumulate):
            # get action from hierarchy
            self.hierarchy.set_input((state_pattern, self.action_pattern))
            self.action_pattern = self.hierarchy.output_block.get_output('feedback')
            self.accumulated_action = np.concatenate([self.accumulated_action, self.action_pattern + shift])
            shift += self.elementary_actions.encoder.output_sdr_size

        # train empowerment tm
        if self.use_intrinsic_reward and (self.empowerment.filename is None):
            current_obs = self.hierarchy.visual_block.get_output('basal')
            if (self.previous_obs.size > 0) and (current_obs.size > 0):
                self.empowerment.learn(self.previous_obs, current_obs)
            self.previous_obs = current_obs

        return self.accumulated_action

    def get_intrinsic_reward(self):
        if self.empowerment.filename is None:
            state = self.hierarchy.visual_block.get_output('basal')
            reward = self.empowerment.eval_state(
                state,
                self.empowerment_horizon,
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

    def backup_agent(self):
        pass

    def restore_agent(self):
        pass
