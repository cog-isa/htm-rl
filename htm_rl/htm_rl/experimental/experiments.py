from htm_rl.agent.agent import Agent, AgentRunner
from htm_rl.agent.memory import Memory, TemporalMemory
from htm_rl.agent.planner import Planner
from htm_rl.common.s_sdr_encoder import StateSDREncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder, format_sa_superposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.envs.gridworld_pomdp import GridWorld
from htm_rl.envs.gridworld_pomdp import MapGenerator

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
    transit_memory=False,
    transit_goals=False
)


class TestRunner:
    def __init__(self, params):
        self.seed = params['seed']
        self.run_param = params['run_param']
        self.planner_param = params['planner_param']
        self.start_indicator = params['start_indicator']
        self.tm_pars = params['tm_pars']
        self.encoder_param = params['encoder_param']
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
        self.transit_memory = params['transit_memory']
        self.transit_goals = params['transit_goals']
        self.test_number = 0

    def run(self, n_tests=None):
        if n_tests is None:
            n_tests = self.n_tests

        for _ in range(n_tests):
            self._init_run()
            self.runner.run()
            self.steps.append(self.runner.train_stats.steps)
            self.env.reset()
            self.maps.append(dict(world=self.env.world_description,
                                  position=self.env.agent_position,
                                  direction=self.env.agent_direction,
                                  string_repr=self.env.render()))
            self.test_number += 1

    def _init_run(self):
        self._init_environment()
        self._init_encoders()
        if not self.transit_memory or self.test_number == 0:
            self._init_memory()
        if not self.transit_goals or self.test_number == 0:
            self._init_planner()
        self._init_agent()
        self._init_agent_runner()

    def _init_environment(self):
        if self.fixed_map is not None:
            map_ = self.fixed_map
            if not self.fixed_reward:
                map_ = self.map_generator.place_reward(map_, self.map_generator.get_seed())
            if self.fixed_pos is None:
                pos = self.map_generator.generate_position(map_, self.map_generator.get_seed())
            else:
                pos = self.fixed_pos
        else:
            map_ = next(self.map_generator)
            pos = self.map_generator.generate_position(map_, self.map_generator.get_seed())

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
