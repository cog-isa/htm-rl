from hima.agents.hima.configurator import configure

import yaml
import sys
import ast
import wandb

if len(sys.argv) > 1:
    default_config = sys.argv[1]
else:
    default_config = 'coppelia/pulse_options_continues'

with open(f'../configs/{default_config}.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.Loader)

if config['log']:
    logger = wandb
else:
    logger = None

for arg in sys.argv[2:]:
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

if logger is not None:
    logger = logger.init(project=config['project'], entity=config['entity'], config=config)

# with open('../../experiments/hima/htm_config_unpacked.yaml', 'w') as file:
#     yaml.dump(configure(config), file, Dumper=yaml.Dumper)
if config['environment_type'] == 'gridworld':
    from hima.agents.hima.runners.gridworld import GwHIMARunner
    runner = GwHIMARunner(configure(config), logger=logger, logger_config=config['logger_config'])
elif config['environment_type'] == 'coppelia':
    from hima.agents.hima.runners.coppelia import ArmHIMARunner
    runner = ArmHIMARunner(configure(config), logger=logger, logger_config=config['logger_config'])
else:
    raise ValueError(
        f"Unknown environment type: {config['environment_type']}"
    )

# if logger is not None:
#     runner.draw_map(logger)

runner.run_episodes()
