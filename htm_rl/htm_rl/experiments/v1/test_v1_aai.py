from htm_rl.agents.v1.runner import AAIPygameRunner


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
    test()