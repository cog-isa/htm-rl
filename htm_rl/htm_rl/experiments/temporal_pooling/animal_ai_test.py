import numpy as np
import matplotlib.pyplot as plt

from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.actions import AAIActions

from htm_rl.agents.agent import Agent
from htm_rl.common.sdr import SparseSdr


class SpinAgent(Agent):
    n_actions: int

    def __init__(self, action: int):
        self.action = action

    @property
    def name(self):
        return 'spin'

    def act(self, reward: float, state: SparseSdr, first: bool):
        return self.action

    def reset(self):
        pass


class SpinRunner:
    def __init__(self, config):
        self.env = AnimalAIEnvironment(**config['environment'])
        self.behavior = list(self.env.behavior_specs.keys())[0]
        self.actions = AAIActions().allActions
        self.steps = config['steps']
        self.agent = SpinAgent(**config['agent'])

    def run(self):
        firststep = True
        for i in range(self.steps):
            if firststep:
                self.env.step()
                firststep = False
                dec, term = self.env.get_steps(self.behavior)

            pos = self.env.get_obs_dict(dec.obs)["position"]
            vel = self.env.get_obs_dict(dec.obs)["velocity"]
            cam = self.env.get_obs_dict(dec.obs)["camera"]
            action = self.agent.act(pos, vel, cam)

            self.env.set_actions(self.behavior, self.actions[action].action_tuple)
            self.env.step()
            dec, term = self.env.get_steps(self.behavior)
            if len(term) > 0:  # Episode is over
                firststep = True
        self.env.close()


def collect_data():
    config = {
        'steps': 1000,
        'environment': {
            'seed': 0,
            'file_name': '/home/ivan/htm/AnimalAI-Olympics/env/AnimalAI.x86_64',
            'arenas_configurations': '/home/artem/projects/animal-ai/configs/basic/1g.yml',
            'play': False,
            'useCamera': True,
            'useRayCasts': False,
            'resolution': 200,
        },
        'agent': {
            'action': 2
        }
    }
    runner = SpinRunner(config)
    runner.run()
    plt.imshow(runner.agent.map / runner.agent.map.max())
    plt.show()
    print(runner.agent.map)
    n_states = int(runner.agent.map.min())
    data = np.empty((25, n_states, 200, 200, 3))
    ind = 0
    for i in range(5):
        for j in range(5):
            arr = runner.agent.data[(i, j)]
            temp = np.arange(len(arr))
            np.random.shuffle(temp)
            data[ind] = np.array(arr)[temp[:n_states]]
            ind += 1
    np.savez_compressed('aii_data.npz', data)


def test():
    simple_configs = [
        {
            'g_kernel_size': 12,
            'g_stride': 2,
            'g_sigma_x': 4,
            'g_sigma_y': 4,
            'g_lambda_': 8,
            'g_filters': 8,
            'activity_level': 0.3
        }
    ]

    complex_config = {
        'g_kernel_size': 12,
        'g_stride': 6,
        'g_sigma': 19.2,
        'activity_level': 0.6
    }

    runner = AAIPygameRunner(
        0,
        '/home/artem/projects/animal-ai/env/AnimalAI',
        '/home/artem/projects/animal-ai/configs/basic/1g.yml',
        200,
        {
            'complex_config': complex_config,
            'simple_configs': simple_configs
        },
        2,
        10
    )
    runner.run()