import yaml
import wandb
import sys
import ast
from htm_rl.agents.pmc.runner import RunnerRAG2D, RunnerArm, RunnerAAI


envs = ['rag2d', 'coppelia', 'aai']

if len(sys.argv) > 1:
    env = sys.argv[1]
else:
    env = envs[1]

if env == envs[0]:
    name = 'configs/base_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = RunnerRAG2D(config)
    runner.run()
elif env == envs[1]:
    def configure(config):
        config['environment']['seed'] = config['seed']
        config['agent']['config']['bg']['seed'] = config['seed']
        config['agent']['config']['pmc']['seed'] = config['seed']
        
        config['agent']['config']['bg']['output_size'] = config['agent']['config']['pmc']['n_neurons']
        return config

    if len(sys.argv) > 2:
        name = sys.argv[2]
    else:
        name = 'ur3_config'

    with open(f'configs/{name}.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    for arg in sys.argv[3:]:
        key, value = arg.split('=')

        value = ast.literal_eval(value)

        key = key.lstrip('-')
        if key.endswith('.'):
            # a trick that allow distinguishing sweep params from config params
            # by adding a suffix `.` to sweep param - now we should ignore it
            key = key[:-1]
        tokens = key.split('.')
        c = config
        for k in tokens[:-1]:
            if not k:
                # a trick that allow distinguishing sweep params from config params
                # by inserting additional dots `.` to sweep param - we just ignore it
                continue
            if 0 in c:
                k = int(k)
            c = c[k]
        c[tokens[-1]] = value

    config = configure(config)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    else:
        logger = None

    runner = RunnerArm(config, logger=logger)
    runner.run()

elif env == envs[2]:
    name = 'configs/aai_basic_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    runner = RunnerAAI(config)
    runner.run()
