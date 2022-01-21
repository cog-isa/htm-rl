import yaml
import wandb
from runner import RunnerRAG2D, RunnerPulse

envs = ['rag2d', 'pulse']
env = envs[1]

if env == envs[0]:
    name = 'configs/base_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    runner = RunnerRAG2D(config)
    runner.run()
else:
    name = 'configs/pulse_config.yaml'
    with open(name, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if config['log']:
        logger = wandb.init(project=config['project'], entity=config['entity'], config=config)
    runner = RunnerPulse(config)
    runner.run()
