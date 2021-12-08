import numpy as np
import htm_rl.modules.v1 as v1
import tqdm

import yaml
import sys
import ast
import wandb

if len(sys.argv) > 1:
    default_config_name = sys.argv[1]
else:
    default_config_name = 'base'
with open(f'{default_config_name}.yaml', 'r') as file:
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

# main part
v1_encoder = v1.V1((256, 256), config['complex'], *config['simple'])


def preprocess(data, size):
    data_new = np.zeros((data.shape[0], size))
    for i, img in tqdm.tqdm(enumerate(data), total=data.shape[0]):
        sparse, dense = v1_encoder.compute(img)
        data_new[i][sparse[0]] = 1
    return data_new


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

x_train_mod = preprocess(x_train, 20*22*22)
x_test_mod = preprocess(x_test, 20*22*22)


