import numpy as np
from pickle import dump, load
from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.actions import AAIActions

from htm_rl.agents.agent import Agent
from htm_rl.common.sdr import SparseSdr
from htm.bindings.sdr import SDR

from htm_rl.modules.v1 import V1


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

    def run(self, pics: list):
        firststep = True
        for i in range(self.steps):
            if firststep:
                self.env.step()
                firststep = False
                dec, term = self.env.get_steps(self.behavior)

            pos = self.env.get_obs_dict(dec.obs)["position"]
            vel = self.env.get_obs_dict(dec.obs)["velocity"]
            cam = self.env.get_obs_dict(dec.obs)["camera"]
            pics.append(cam)
            action = self.agent.act(pos, vel, cam)

            self.env.set_actions(self.behavior, self.actions[action].action_tuple)
            self.env.step()
            dec, term = self.env.get_steps(self.behavior)
            if len(term) > 0:  # Episode is over
                firststep = True
        self.env.close()


def list_to_np(lst: list) -> np.ndarray:
    res = np.empty((len(lst), *lst[0].shape))
    for i, arr in enumerate(lst):
        res[i] = arr
    return res


def collect_data():
    config = {
        'steps': 60,
        'environment': {
            'seed': 0,
            'file_name': '/home/ivan/htm/AnimalAI-Olympics/animal-ai/env/AnimalAI',
            'arenas_configurations':
                '/home/ivan/htm/htm-rl/htm_rl/htm_rl/experiments/temporal_pooling/configs/arenas/1g.yml',
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
    pics = []
    runner.run(pics)
    return pics


def through_v1(images: np.ndarray, v1_config):
    resolution = images.shape[-2]
    v1 = V1((resolution, resolution), v1_config['complex_config'], *v1_config['simple_configs'])
    result = []
    for img in images:
        result.append(v1.compute(img)[0][0])
    return {
        'shape': (resolution, resolution),
        'sparse': result
    }

# def test():
#     simple_configs = [
#         {
#             'g_kernel_size': 12,
#             'g_stride': 2,
#             'g_sigma_x': 4,
#             'g_sigma_y': 4,
#             'g_lambda_': 8,
#             'g_filters': 8,
#             'activity_level': 0.3
#         }
#     ]
#
#     complex_config = {
#         'g_kernel_size': 12,
#         'g_stride': 6,
#         'g_sigma': 19.2,
#         'activity_level': 0.6
#     }

#     runner = AAIPygameRunner(
#         0,
#         '/home/artem/projects/animal-ai/env/AnimalAI',
#         '/home/artem/projects/animal-ai/configs/basic/1g.yml',
#         200,
#         {
#             'complex_config': complex_config,
#             'simple_configs': simple_configs
#         },
#         2,
#         10
#     )
#     runner.run()


if __name__ == '__main__':
    pics = collect_data()
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
    converted = through_v1(list_to_np(pics), {
        'simple_configs': simple_configs,
        'complex_config': complex_config
    })
    print(converted['sparse'][0])
    print(type(converted['sparse'][0]))
    with open('saved_dictionary.pkl', 'wb') as f:
        dump(converted, f)

