from htm_rl.agent.agent import Agent, AgentRunner
from htm_rl.agent.memory import Memory, TemporalMemory
from htm_rl.agent.planner import Planner
from htm_rl.common.s_sdr_encoder import StateSDREncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder, format_sa_superposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.envs.gridworld_pomdp import GridWorld
from htm_rl.envs.gridworld_pomdp import MapGenerator
from htm_rl.baselines.dqn_agent import DqnAgent, DqnAgentRunner
from htm_rl.agent.train_eval import RunResultsProcessor
import os

default_parameters = dict(
    tm_pars=dict(n_columns=None,
                 cells_per_column=1,
                 activation_threshold=float(1.0),
                 learning_threshold=float(1.0),
                 initial_permanence=0.6,
                 connected_permanence=0.5,
                 maxNewSynapseCount=float(1.0),
                 maxSynapsesPerSegment=float(1.0),
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.025,
                 predictedSegmentDecrement=0.005),
    run_param=dict(
        n_episodes=300,
        max_steps=25,
        pretrain=50,
        verbosity=0
    ),
    planner_param=dict(
        planning_horizon=15,
        goal_memory_size=5,
        alpha=0.2
    ),
    encoder_param=dict(
        action=dict(value_bits=5,
                    activation_threshold=5),
        state=dict(threshold=1.0)
    ),
    env_param=dict(
        max_sight_range=None,
        observable_vars=['window'],
        window_coords={'top_left': (1, -1),
                       'bottom_right': (0, 1)}
    ),
    dqn_param=dict(epsilon=0.15,
                   gamma=0.975,
                   learning_rate=.3e-3),
    seed=1,
    start_indicator=None,
    fixed_direction=None,
    fixed_pos=None,
    fixed_reward=False,
    fixed_map=None,
    n_tests=3,
    verbosity=1,
    shape=(3, 3),
    complexity=0.75,
    density=0.75,
    transfer_memory=False,
    transfer_goals=False,
    test_name='agent_test',
    agent_type='HTM',
    test_dir='agent_test',
    moving_average=20,
)


class TestRunner:
    def __init__(self, params):
        self.agent_type = params['agent_type']
        self.test_name = params['test_name']
        self.test_dir = params['test_dir']
        try:
            os.mkdir(self.test_dir)
        except FileExistsError:
            pass
        self.moving_average = params['moving_average']
        self.seed = params['seed']
        self.run_param = params['run_param']
        self.env_param = params['env_param']
        self.fixed_direction = params['fixed_direction']
        self.fixed_pos = params['fixed_pos']
        self.fixed_reward = params['fixed_reward']
        self.fixed_map = params['fixed_map']
        self.n_tests = params['n_tests']
        self.steps = list()
        self.maps = list()
        self.verbosity = params['verbosity']
        self.map_generator = MapGenerator(shape=params['shape'],
                                          complexity=params['complexity'],
                                          density=params['density'],
                                          s=self.seed)
        self.transfer_memory = params['transfer_memory']
        self.test_number = 0

        if self.agent_type == 'HTM':
            self.planner_param = params['planner_param']
            self.start_indicator = params['start_indicator']
            self.tm_pars = params['tm_pars']
            self.encoder_param = params['encoder_param']
            self.transfer_goals = params['transfer_goals']
            self.agent_name = f'htm_{self.planner_param["planning_horizon"]}_{self.planner_param["goal_memory_size"]}g'
        elif self.agent_type == 'DQN':
            self.dqn_param = params['dqn_param']
            self.agent_name = 'dqn'
        else:
            raise NotImplemented

        self.results_processor = RunResultsProcessor(self.test_name,
                                                     self.test_dir,
                                                     self.moving_average,
                                                     self.verbosity)
        self.map_seed = 0

    def run(self, n_tests=None):
        if n_tests is None:
            n_tests = self.n_tests

        if self.agent_type == 'HTM':
            init = self._init_run
        elif self.agent_type == 'DQN':
            init = self._init_dqn_run
        else:
            raise NotImplemented

        for _ in range(n_tests):
            init()
            self.runner.run()
            # gather data
            self.results_processor.store_result(self.runner.train_stats,
                                                self.agent_name + f'_{self.test_number}')
            self.env.reset()
            self.maps.append((self.env.world_description,
                              (self.env.agent_position['row'],
                               self.env.agent_position['column']),
                              self.env.agent_direction,
                              self.map_seed))
            if self.verbosity >= 2:
                print(self.env.render())

            self.results_processor.store_environment_maps([])

            self.test_number += 1

        self.results_processor.store_environment_maps(self.maps)

    def _init_dqn_agent(self):
        self.agent = DqnAgent(self.env.window_size[0] * self.env.window_size[1],
                              self.env.n_actions,
                              seed=self.seed,
                              **self.dqn_param)

    def _init_dqn_runner(self):
        self.runner = DqnAgentRunner(agent=self.agent,
                                     env=self.env,
                                     **self.run_param)

    def _init_dqn_run(self):
        self._init_environment()
        if not self.transfer_memory or self.test_number == 0:
            self._init_dqn_agent()
        self._init_dqn_runner()

    def _init_run(self):
        self._init_environment()
        self._init_encoders()
        if not self.transfer_memory or self.test_number == 0:
            self._init_memory()
        if not self.transfer_goals or self.test_number == 0:
            self._init_planner()
        self._init_agent()
        self._init_agent_runner()

    def _init_environment(self):
        s = self.map_generator.get_seed()
        if self.fixed_map is not None:
            map_ = self.fixed_map
            if not self.fixed_reward:
                map_ = self.map_generator.place_reward(map_, s)
            if self.fixed_pos is None:
                pos = self.map_generator.generate_position(map_, s)
            else:
                pos = self.fixed_pos
        else:
            map_ = next(self.map_generator)
            pos = self.map_generator.generate_position(map_, s)

        if self.fixed_direction is None:
            direction = self.map_generator.random_gen.integers(3)
        else:
            direction = self.fixed_direction

        self.env = GridWorld(world_description=map_,
                             agent_initial_position={'row': int(pos[0]),
                                                     'column': int(pos[1])},
                             agent_initial_direction=direction,
                             world_size=self.map_generator.shape,
                             **self.env_param)
        self.map_seed = s

    def _init_encoders(self):
        action_encoder_param = self.encoder_param['action']
        self.action_encoder = IntSdrEncoder(name='action', n_values=self.env.n_actions, **action_encoder_param)
        state_encoder_param = self.encoder_param['state']
        self.state_encoder = StateSDREncoder(name='state', n_values=self.env.dimensions['surface']+1,
                                             shape=self.env.window_size,
                                             **state_encoder_param)

        self.sa_encoder = SaSdrEncoder(self.state_encoder, self.action_encoder)

    def _init_memory(self):
        if self.test_number == 0:
            self.tm_pars['n_columns'] = self.sa_encoder.total_bits
            if isinstance(self.tm_pars['activation_threshold'], float):
                self.tm_pars['activation_threshold'] *= self.sa_encoder.value_bits
                self.tm_pars['activation_threshold'] = int(self.tm_pars['activation_threshold'])
            elif self.tm_pars['activation_threshold'] < 0:
                self.tm_pars['activation_threshold'] += self.sa_encoder.value_bits

            if isinstance(self.tm_pars['learning_threshold'], float):
                self.tm_pars['learning_threshold'] *= self.sa_encoder.value_bits
                self.tm_pars['learning_threshold'] = int(self.tm_pars['learning_threshold'])
            elif self.tm_pars['learning_threshold'] < 0:
                self.tm_pars['learning_threshold'] += self.sa_encoder.value_bits

            if isinstance(self.tm_pars['maxNewSynapseCount'], float):
                self.tm_pars['maxNewSynapseCount'] *= self.sa_encoder.value_bits
                self.tm_pars['maxNewSynapseCount'] = int(self.tm_pars['maxNewSynapseCount'])
            elif self.tm_pars['maxNewSynapseCount'] < 0:
                self.tm_pars['maxNewSynapseCount'] += self.sa_encoder.value_bits

            if isinstance(self.tm_pars['maxSynapsesPerSegment'], float):
                self.tm_pars['maxSynapsesPerSegment'] *= self.sa_encoder.value_bits
                self.tm_pars['maxSynapsesPerSegment'] = int(self.tm_pars['maxSynapsesPerSegment'])
            elif self.tm_pars['maxSynapsesPerSegment'] < 0:
                self.tm_pars['maxSynapsesPerSegment'] += self.sa_encoder.value_bits

        tm = TemporalMemory(**self.tm_pars)
        self.memory = Memory(tm=tm, encoder=self.sa_encoder, sdr_formatter=self.sa_encoder.format,
                             sa_superposition_formatter=format_sa_superposition,
                             start_indicator=self.start_indicator)

    def _init_planner(self):
        self.planner = Planner(memory=self.memory, state_encoder=self.state_encoder, **self.planner_param)

    def _init_agent(self):
        self.agent = Agent(memory=self.memory, planner=self.planner, n_actions=self.env.n_actions)

    def _init_agent_runner(self):
        self.runner = AgentRunner(agent=self.agent, env=self.env,
                                  **self.run_param)