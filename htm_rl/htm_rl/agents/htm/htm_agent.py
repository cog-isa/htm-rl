from htm_rl.agents.htm.hierarchy import Hierarchy, Block, InputBlock, SpatialMemory
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.agents.htm.basal_ganglia import BasalGanglia2, BasalGanglia, BasalGanglia3
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.agents.htm.configurator import configure
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm.bindings.sdr import SDR
import imageio
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
import wandb


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


class HTMAgentRunner:
    def __init__(self, config):
        seed = config['seed']
        np.random.seed(seed)
        random.seed(seed)

        block_configs = config['blocks']
        input_block_configs = config['input_blocks']

        blocks = list()

        for block_conf in input_block_configs:
            blocks.append(InputBlock(**block_conf))

        for block_conf in block_configs:
            tm = ApicalBasalFeedbackTM(**block_conf['tm'])

            sm = SpatialMemory(**block_conf['sm'])

            if block_conf['sp'] is not None:
                sp = SpatialPooler(**block_conf['sp'])
            else:
                sp = None

            if block_conf['bg'] is not None:
                if block_conf['bg_sp'] is not None:
                    bg_sp = SpatialPooler(**block_conf['bg_sp'])
                else:
                    bg_sp = None
                if config['basal_ganglia_version'] == 1:
                    bg = BasalGanglia(sp=bg_sp, **block_conf['bg'])
                elif config['basal_ganglia_version'] == 2:
                    bg = BasalGanglia2(sp=bg_sp, **block_conf['bg'])
                elif config['basal_ganglia_version'] == 3:
                    bg = BasalGanglia3(sp=bg_sp, **block_conf['bg'])
                else:
                    raise ValueError('There is no such version of BG')
            else:
                bg = None

            blocks.append(Block(tm=tm, sm=sm, sp=sp, bg=bg, **block_conf['block']))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])

        self.agent = HTMAgent(config['agent'], hierarchy)
        self.environment = BioGwLabEnvironment(**config['environment'])

    def run_episodes(self, n_episodes, train_patterns=True, logger=None, log_q_table=False, log_every_episode=50, log_patterns=False):
        history = {'steps': list(), 'reward': list()}

        total_reward = 0
        steps = 0
        episode = -1
        animation = False
        prev_reward = 0
        while episode < n_episodes:
            if train_patterns:
                self.agent.train_patterns()

            reward, obs, is_first = self.environment.observe()

            self.agent.reinforce(prev_reward)
            prev_reward = reward

            if is_first:
                self.agent.reset()

                if animation:
                    animation = False
                    with imageio.get_writer(f'/tmp/steps_{episode}.gif', mode='I') as writer:
                        for i in range(steps):
                            image = imageio.imread(f'/tmp/step_{i}.png')
                            writer.append_data(image)
                    logger.log({f'animation': logger.Video(f'/tmp/steps_{episode}.gif', fps=4, format='gif')})

                history['steps'].append(steps)
                history['reward'].append(total_reward)

                if (logger is not None) and (episode > -1):
                    logger.log({'steps': steps, 'reward': total_reward, 'episode': episode})

                episode += 1
                steps = 0
                total_reward = 0

                if ((episode % log_every_episode) == 0) and (logger is not None):
                    if log_patterns:
                        if log_q_table:
                            q = self.agent.hierarchy.output_block.bg.input_weights_d1 - self.agent.hierarchy.output_block.bg.input_weights_d2
                            table = list()
                        grid_shape = self.environment.env.shape
                        n_states = grid_shape[0] * grid_shape[1]
                        state_encoder = IntBucketEncoder(n_states, 5)
                        sp_state = SDR(self.agent.hierarchy.blocks[2].tm.basal_columns)
                        sp_input = SDR(state_encoder.output_sdr_size)
                        sp_action_input = SDR(self.agent.muscles.muscles_size)
                        sp_action = SDR(self.agent.hierarchy.output_block.tm.basal_columns)
                        states_patterns = np.zeros((n_states, sp_state.size))
                        action_patterns = np.zeros((self.agent.action.n_actions-1, sp_action.size))
                        action_sparse_patterns = list()

                        for j in range(self.agent.action.n_actions-1):
                            sp_action_input.dense = self.agent.action.patterns[j]
                            self.agent.hierarchy.output_block.sp.compute(sp_action_input, False, sp_action)
                            action_sparse_patterns.append(np.copy(sp_action.sparse))
                            action_patterns[j, sp_action.sparse] = 1

                        for i in range(n_states):
                            state = state_encoder.encode(i)
                            sp_input.sparse = state
                            self.agent.hierarchy.blocks[2].sp.compute(sp_input, False, sp_state)
                            states_patterns[i, sp_state.sparse] = 1
                            row = [str((i // grid_shape[0], i % grid_shape[1]))]
                            if log_q_table:
                                for j, option in enumerate(action_sparse_patterns):
                                    row.append(str(round(q[option][:, sp_state.sparse].mean(), 4)))
                                table.append(row)
                        if log_q_table:
                            logger.log({f'table': logger.Table(data=table, columns=['state', 'right', 'down', 'left', 'up'])})

                        plt.imsave(f'/tmp/states_{config["seed"]}.png', states_patterns)
                        logger.log({'states': wandb.Image(f'/tmp/states_{config["seed"]}.png', )})

                        plt.imsave(f'/tmp/sm_options_{config["seed"]}.png', self.agent.hierarchy.output_block.sm.patterns)
                        logger.log({'sm options': wandb.Image(f'/tmp/sm_options_{config["seed"]}.png', )})

                        plt.imsave(f'/tmp/actions_{config["seed"]}.png', action_patterns)
                        logger.log({'actions': wandb.Image(f'/tmp/actions_{config["seed"]}.png', )})

                        action_intersection = np.dot(action_patterns, action_patterns.T)
                        state_intersection = np.dot(states_patterns, states_patterns.T)

                        logger.log({
                            'action_intersection': logger.Table(data=action_intersection,
                                                                        columns=[str(x) for x in range(action_patterns.shape[0])]),
                            'state_intersection': logger.Table(data=state_intersection,
                                                                       columns=[str(x) for x in range(states_patterns.shape[0])])
                                    })

                    animation = True
            else:
                steps += 1
                total_reward += reward

            if animation:
                plt.imsave(f'/tmp/step_{steps}.png', self.environment.callmethod('render_rgb'))

            action = self.agent.make_action(obs)
            self.environment.act(action)
        return history


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        default_config_name = sys.argv[1]
    else:
        default_config_name = 'two_levels_16x16_obs_default'
    with open(f'../../experiments/htm_agent/{default_config_name}.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    wandb.init(config=config)

    for arg in sys.argv[2:]:
        key, value = arg.split('=')

        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    value = [int(value.strip('[]'))]

        key = key.lstrip('-')
        tokens = key.split('.')
        if len(tokens) == 4:
            config[tokens[0]][int(tokens[1])][tokens[2]][tokens[3]] = value
        elif len(tokens) == 2:
            config[tokens[0]][tokens[1]] = value
        elif len(tokens) == 1:
            config[tokens[0]] = value

    # with open('../../experiments/htm_agent/htm_config_unpacked.yaml', 'w') as file:
    #     yaml.dump(configure(config), file, Dumper=yaml.Dumper)

    runner = HTMAgentRunner(configure(config))
    runner.agent.train_patterns()

    plt.imsave(f'/tmp/map_{config["environment"]["seed"]}.png', runner.environment.callmethod('render_rgb'))
    wandb.log({'map': wandb.Image(f'/tmp/map_{config["environment"]["seed"]}.png',)})

    if config['basal_ganglia_version'] == 1:
        log_q_table = False
    else:
        log_q_table = True

    history = runner.run_episodes(500, logger=wandb, log_q_table=False, log_every_episode=50, log_patterns=False)

    wandb.log({'av_steps': np.array(history['steps']).mean()})

