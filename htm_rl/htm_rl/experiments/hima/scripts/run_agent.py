from htm_rl.agents.hima.runner import HIMAgentRunner
from htm_rl.agents.hima.configurator import configure

import yaml
import sys
import ast
import wandb

if len(sys.argv) > 1:
    default_config_name = sys.argv[1]
else:
    default_config_name = 'pulse_options'
with open(f'../configs/{default_config_name}.yaml', 'r') as file:
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

runner = HIMAgentRunner(configure(config), logger=logger)

# if logger is not None:
#     runner.draw_map(logger)

runner.run_episodes(**config['run_options'])
