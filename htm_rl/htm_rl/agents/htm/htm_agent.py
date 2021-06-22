from htm_rl.agents.htm.hierarchy import Hierarchy, Block, InputBlock, SpatialMemory
from htm_rl.agents.htm.muscles import Muscles
from htm_rl.modules.basal_ganglia import BasalGanglia
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.agents.htm.configurator import configure
from htm.bindings.algorithms import SpatialPooler
from htm_rl.agents.htm.htm_apical_basal_feeedback import ApicalBasalFeedbackTM
from htm_rl.agents.htm.utils import OptionVis, draw_values, compute_q_policy, draw_policy
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
        self.use_intrinsic_reward = config['intrinsic_reward']
        self.theta = config['theta']
        self.punish_reward = config['punish_reward']
        self.hierarchy = hierarchy
        self.action = BioGwLabAction(**config['action'])

        self.muscles = Muscles(**config['muscles'])

        self.total_patterns = 2 ** self.action.muscles_size
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

    def get_intrinsic_reward(self):
        confidence = self.hierarchy.blocks[2].tm.confidence[-2]
        conf_threshold = self.hierarchy.blocks[2].confidence_threshold
        anomaly = self.hierarchy.blocks[2].tm.anomaly[-1]
        anomaly_threshold = self.hierarchy.blocks[2].tm.anomaly_threshold
        if (anomaly > anomaly_threshold) and (confidence > conf_threshold):
            reward = anomaly - anomaly_threshold
        else:
            reward = 0
        return reward

    def reinforce(self, reward, is_terminal=False):
        """
        Reinforce BasalGanglia.
        :param reward: float:
        Main reward of the environment.
        Rewards for all blocks in hierarchy, they may differ from actual reward.
        List should be length of number of blocks in hierarchy.
        :param is_terminal:
        :return:
        """
        if self.use_intrinsic_reward:
            reward += (self.theta * self.get_intrinsic_reward())

        self.hierarchy.add_rewards([reward] * len(self.hierarchy.blocks))

        if is_terminal:
            for block in self.hierarchy.blocks:
                block.reinforce(is_terminal=is_terminal)
        else:
            self.hierarchy.output_block.reinforce()

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
                bg = BasalGanglia(**block_conf['bg'])
            else:
                bg = None

            blocks.append(Block(tm=tm, sm=sm, sp=sp, bg=bg, **block_conf['block']))

        hierarchy = Hierarchy(blocks, **config['hierarchy'])

        if 'scenario' in config.keys():
            self.scenario = Scenario(config['scenario'], self)
        else:
            self.scenario = None
        self.agent = HTMAgent(config['agent'], hierarchy)
        self.env_config = config['environment']
        self.environment = BioGwLabEnvironment(**config['environment'])
        if 'positions' in config['environment']['food'].keys():
            self.terminal_positions = [tuple(x) for x in config['environment']['food']['positions']]
            self.terminal_pos_stat = dict(zip(self.terminal_positions, [0] * len(self.terminal_positions)))
        else:
            self.terminal_positions = None
            self.terminal_pos_stat = None

        self.total_reward = 0
        self.animation = False
        self.agent_pos = list()
        self.steps = 0
        self.episode = 0
        self.option_actions = list()
        self.option_predicted_actions = list()
        self.current_option_id = None
        self.current_action = None

        self.option_stat = OptionVis(self.environment.env.shape, **config['vis_options'])
        self.option_start_pos = None
        self.option_end_pos = None

        self.seed = seed
        self.rng = random.Random(self.seed)

    def run_episodes(self, n_episodes, train_patterns=True, logger=None, log_values=False, log_policy=False,
                     log_every_episode=50,
                     log_segments=False, draw_options=False, log_terminal_stat=False, draw_options_stats=False,
                     opt_threshold=0):
        self.total_reward = 0
        self.steps = 0
        self.episode = 0
        self.animation = False
        self.agent_pos = list()

        while self.episode < n_episodes:
            if self.scenario is not None:
                self.scenario.check_conditions()

            if train_patterns:
                self.agent.train_patterns()

            reward, obs, is_first = self.environment.observe()

            if is_first:
                # ///logging///
                if self.animation:
                    # log all saved frames for this episode
                    self.animation = False
                    with imageio.get_writer(f'/tmp/{logger.run.id}_episode_{self.episode}.gif', mode='I', fps=2) as writer:
                        for i in range(self.steps):
                            image = imageio.imread(f'/tmp/{logger.run.id}_episode_{self.episode}_step_{i}.png')
                            writer.append_data(image)
                    logger.log({f'animation': logger.Video(f'/tmp/{logger.run.id}_episode_{self.episode}.gif', fps=2,
                                                           format='gif')}, step=self.episode)

                if (logger is not None) and (self.episode > 0):
                    logger.log({'steps': self.steps, 'reward': self.total_reward, 'episode': self.episode}, step=self.episode)
                    if log_segments:
                        logger.log(
                            {'basal_segments': self.agent.hierarchy.output_block.tm.basal_connections.numSegments(),
                             'apical_segments': self.agent.hierarchy.output_block.tm.apical_connections.numSegments(),
                             'exec_segments': self.agent.hierarchy.output_block.tm.exec_feedback_connections.numSegments(),
                             'inhib_segments': self.agent.hierarchy.output_block.tm.inhib_connections.numSegments(),
                             'basal_syn': self.agent.hierarchy.output_block.tm.basal_connections.numSynapses(),
                             'apical_syn': self.agent.hierarchy.output_block.tm.apical_connections.numSynapses(),
                             'exec_syn': self.agent.hierarchy.output_block.tm.exec_feedback_connections.numSynapses(),
                             'inhib_syn': self.agent.hierarchy.output_block.tm.inhib_connections.numSynapses()
                             },
                            step=self.episode)

                if ((self.episode % log_every_episode) == 0) and (logger is not None) and (self.episode > 0):
                    if draw_options_stats:
                        self.option_stat.draw_options(logger, self.episode, threshold=opt_threshold,
                                                      obstacle_mask=self.environment.env.entities['obstacle'].mask)
                        self.option_stat.clear_stats()
                    if log_terminal_stat and (self.terminal_pos_stat is not None):
                        logger.log(dict([(str(x[0]), x[1]) for x in self.terminal_pos_stat.items()]), step=self.episode)
                    if log_values or log_policy:
                        if len(self.option_stat.action_displace) == 3:
                            directions = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
                            actions_map = {0: 'move', 1: 'turn_right', 2: 'turn_left'}
                        else:
                            directions = None
                            actions_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

                        q, policy, actions = compute_q_policy(self.environment.env, self.agent, directions)

                        if log_values:
                            draw_values(f'/tmp/values_{logger.run.id}_{self.episode}.png',
                                        self.environment.env.shape,
                                        q,
                                        policy,
                                        directions=directions)
                            logger.log({'state_values': wandb.Image(f'/tmp/values_{logger.run.id}_{self.episode}.png')},
                                       step=self.episode)
                        if log_policy:
                            draw_policy(f'/tmp/policy_{logger.run.id}_{self.episode}.png',
                                        self.environment.env.shape,
                                        policy,
                                        actions,
                                        directions=directions,
                                        actions_map=actions_map)
                            logger.log({'policy': wandb.Image(f'/tmp/policy_{logger.run.id}_{self.episode}.png')},
                                       step=self.episode)

                if ((((self.episode + 1) % log_every_episode) == 0) or (self.episode == 0)) and (logger is not None):
                    self.animation = True
                    self.agent_pos.clear()
                # \\\logging\\\

                self.agent.reset()

                self.episode += 1
                self.steps = 0
                self.total_reward = 0
            else:
                self.steps += 1
                self.total_reward += reward

            is_terminal = self.environment.callmethod('is_terminal')
            if not is_terminal:
                action = self.agent.make_action(obs)
            else:
                self.agent.hierarchy.output_block.made_decision = True
                action = 0

            self.current_action = action

            self.agent.reinforce(reward, is_terminal=is_terminal)

            # ///logging///
            if draw_options_stats:
                self.update_option_stats()

            if self.animation:
                self.draw_animation_frame(logger, draw_options, self.agent_pos, self.episode, self.steps)
            # \\\logging\\\

            self.environment.act(action)

            # ///logging///
            if self.terminal_pos_stat is not None:
                if self.environment.callmethod('is_terminal'):
                    pos = self.environment.env.agent.position
                    if pos in self.terminal_pos_stat:
                        self.terminal_pos_stat[pos] += 1
            # \\\logging\\\

    def draw_animation_frame(self, logger, draw_options, agent_pos, episode, steps):
        pic = self.environment.callmethod('render_rgb')
        if isinstance(pic, list):
            pic = pic[0]

        if draw_options:
            c_pos = self.environment.env.agent.position
            c_direction = self.environment.env.agent.view_direction

            if self.agent.hierarchy.blocks[5].made_decision:
                agent_pos.append(c_pos)
                if len(agent_pos) > 1:
                    pic[tuple(zip(*agent_pos))] = [[255, 255, 150]] * len(agent_pos)
                else:
                    pic[agent_pos[0]] = [255, 255, 255]
            else:
                if len(agent_pos) > 0:
                    agent_pos.clear()

            term_draw_options = np.zeros((pic.shape[0], 3, 3))
            c_option = self.agent.hierarchy.blocks[5].current_option
            f_option = self.agent.hierarchy.blocks[5].failed_option
            comp_option = self.agent.hierarchy.blocks[5].completed_option
            self.agent.hierarchy.blocks[5].failed_option = None
            self.agent.hierarchy.blocks[5].completed_option = None

            if c_option is not None:
                term_draw_options[c_option, 0] = [255, 255, 255]
            if f_option is not None:
                term_draw_options[f_option, 1] = [200, 0, 0]
            if comp_option is not None:
                term_draw_options[comp_option, 2] = [0, 0, 200]

            if self.agent.hierarchy.output_block.predicted_options is not None:
                for o in self.agent.hierarchy.output_block.predicted_options:
                    predicted_action_pattern = np.flatnonzero(self.agent.hierarchy.output_block.sm.patterns[o])
                    self.agent.muscles.set_active_input(predicted_action_pattern)
                    self.agent.muscles.depolarize_muscles()
                    action_pattern = self.agent.muscles.get_depolarized_muscles()
                    # convert muscles activation pattern to environment action
                    p_action = self.agent.action.get_action(action_pattern)
                    direction = c_direction - self.option_stat.action_rotation[p_action]
                    if direction < 0:
                        direction = 4 - direction
                    else:
                        direction %= 4
                    if (len(self.option_stat.action_displace) == 4) or (np.all(self.option_stat.action_displace[p_action] == 0)):
                        displacement = self.option_stat.action_displace[p_action]
                    else:
                        displacement = self.option_stat.transform_displacement((0, 1), direction)
                    p_pos = (c_pos[0] + displacement[0], c_pos[1] + displacement[1])

                    if (p_pos[0] < pic.shape[0]) and (p_pos[1] < pic.shape[1]):
                        pic[p_pos[0], p_pos[1]] = [255, 200, 120]
            pic = np.concatenate([pic, term_draw_options], axis=1)

        plt.imsave(f'/tmp/{logger.run.id}_episode_{episode}_step_{steps}.png', pic.astype('uint8'))

    def update_option_stats(self):
        option_block = self.agent.hierarchy.blocks[5]

        if self.agent.hierarchy.blocks[5].made_decision:
            if len(self.option_actions) == 0:
                self.option_start_pos = self.environment.env.agent.position
            current_option = option_block.current_option
            current_option_id = option_block.sm.unique_id[current_option]
            if (self.current_option_id != current_option_id) and (len(self.option_actions) != 0):
                # update stats
                self.option_end_pos = self.environment.env.agent.position
                self.option_stat.update(self.current_option_id,
                                        self.option_start_pos,
                                        self.option_end_pos,
                                        self.option_actions,
                                        self.option_predicted_actions)
                self.option_actions.clear()
                self.option_predicted_actions = list()
            if self.agent.hierarchy.blocks[5].made_decision:
                predicted_actions = list()
                if self.agent.hierarchy.output_block.predicted_options is not None:
                    for o in self.agent.hierarchy.output_block.predicted_options:
                        predicted_action_pattern = np.flatnonzero(self.agent.hierarchy.output_block.sm.patterns[o])
                        self.agent.muscles.set_active_input(predicted_action_pattern)
                        self.agent.muscles.depolarize_muscles()
                        action_pattern = self.agent.muscles.get_depolarized_muscles()
                        # convert muscles activation pattern to environment action
                        a = self.agent.action.get_action(action_pattern)
                        predicted_actions.append(a)

                self.option_actions.append(self.current_action)
                self.option_predicted_actions.append(predicted_actions)
                self.current_option_id = current_option_id
        else:
            if len(self.option_actions) > 0:
                if option_block.failed_option is not None:
                    last_option = option_block.failed_option
                elif option_block.completed_option is not None:
                    last_option = option_block.completed_option
                else:
                    last_option = None
                if last_option is not None:
                    last_option_id = option_block.sm.unique_id[last_option]
                    self.option_end_pos = self.environment.env.agent.position
                    self.option_stat.update(last_option_id,
                                            self.option_start_pos,
                                            self.option_end_pos,
                                            self.option_actions,
                                            self.option_predicted_actions)
                self.option_actions.clear()
                self.option_predicted_actions = list()

    def set_food_positions(self, positions, rand=False, sample_size=1):
        if rand:
            positions = self.rng.sample(positions, sample_size)
        self.environment.env.modules['food'].generator.positions = positions


class Scenario:
    def __init__(self, path, runner: HTMAgentRunner):
        self.runner = runner
        self.events = list()
        with open(path, 'r') as file:
            events = yaml.load(file, Loader=yaml.Loader)
            for event in events:
                condition = event['condition']
                action = event['action']
                params = event['params']
                self.events.append({'condition': condition, 'action': action, 'params': params, 'done': False})

    def check_conditions(self):
        for event in self.events:
            attr_name, val = event['condition']
            attr = self.get_attr(attr_name)
            if (attr == val) and (not event['done']):
                method_name = event['action']
                params = event['params']
                f = self.get_attr(method_name)
                f(**params)
                event['done'] = True

    def get_attr(self, attr):
        obj = self.runner
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        default_config_name = sys.argv[1]
    else:
        default_config_name = 'four_rooms_9x9_default'
    with open(f'../../experiments/htm_agent/{default_config_name}.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb
    else:
        logger = None

    if logger is not None:
        logger.init(project=config['project'], entity=config['entity'], config=config)

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

    if logger is not None:
        map_image = runner.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        plt.imsave(f'/tmp/map_{config["environment"]["seed"]}.png', map_image.astype('uint8'))
        logger.log({'map': wandb.Image(f'/tmp/map_{config["environment"]["seed"]}.png', )})

    runner.run_episodes(logger=logger, **config['run_options'])
