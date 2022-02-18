import yaml
import wandb
from runner import RunnerRAG2D, RunnerArm

envs = ['rag2d', 'coppelia']
env = envs[1]

if env == envs[0]:
    name = 'configs/base_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = RunnerRAG2D(config)
    runner.run()
else:
    name = 'configs/ur3_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    runner = RunnerArm(config)
    runner.run()
