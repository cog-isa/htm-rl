import yaml
import wandb
from htm_rl.agents.pmc.runner import RunnerRAG2D, RunnerArm, RunnerAAI

envs = ['rag2d', 'coppelia', 'aai']
env = envs[2]

if env == envs[0]:
    name = 'configs/base_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = RunnerRAG2D(config)
    runner.run()
elif env == envs[1]:
    name = 'configs/ur3_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    runner = RunnerArm(config)
    runner.run()
elif env == envs[2]:
    name = 'configs/aai_basic_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    runner = RunnerAAI(config)
    runner.run()
