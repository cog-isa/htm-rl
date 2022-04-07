from hima.agents.v1.runner import AAIPygameRunner, ExplorationAAIRunner
import matplotlib.pyplot as plt
import numpy as np


def collect_data():
    config = {
        'steps': 1000,
        'environment':{
            'seed': 0,
            'file_name': '/home/artem/projects/animal-ai/env/AnimalAI',
            'arenas_configurations': '/home/artem/projects/animal-ai/configs/basic/1g.yml',
            'play': False,
            'useCamera': True,
            'useRayCasts': False,
            'resolution': 200,
        },
        'agent':{
            'grid_size': 5
        }
    }
    runner = ExplorationAAIRunner(config)
    runner.run()
    plt.imshow(runner.agent.map / runner.agent.map.max())
    plt.show()
    print(runner.agent.map)
    n_states = int(runner.agent.map.min())
    if n_states == 0:
        print("One of the cells is not visited")
        print("No data will be saved")
    else:
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


if __name__ == "__main__":
    # test()
    collect_data()