# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Installation](#installation)
    - [Python package requirements](#python-package-requirements)
    - [[Optional] Using `htm_rl` package in Jupyter Notebooks](#optional-using-htm_rl-package-in-jupyter-notebooks)
  - [Repository structure](#repository-structure)
  - [Quick intro](#quick-intro)
    - [HTM](#htm)
- [Visualization](https://github.com/cog-isa/htm-rl/blob/master/watcher)

## Links

- This repo [link](https://github.com/cog-isa/htm-rl)
- Installation [guide](./install.md) for contributors
- Cumulative project's [readme](./htm_rl/htm_rl/README.md)
- Project releases [folder](./reports) with version-specific reports.
- Project's live [log](./log.md)

## Installation

There're two setup guides:

- [short version](#installation) is down there for users who want just to try out `htm_rl` library
- [complete version](./install.md) for developers wanting to contribute to the project

### Python package requirements

1. `[Optional]` Create new environment (e.g. *htm_rl*) with _conda_ or any alternative like _pipenv_ or _virtualenv_:
  
    ```bash
    conda create --name htm_rl
    conda activate htm_rl
    ```

2. Install requirements specified in _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

### [Optional] Using `htm_rl` package in Jupyter Notebooks

To have an ability to import and use `htm_rl` package in Jupyter Notebooks, install it with _development mode_ flag:

```bash
cd <project_root>/htm_rl

pip install -e .
```

Now you can import modules from the `htm_rl` package:

```python
from pathlib import Path
from htm_rl.config import TestRunner, read_config

config_path = Path('./config.yml')
config = read_config(config_path, verbose=False)

runner = TestRunner(config)
runner.run('dqn', run=True, aggregate=True)
```

## Repository structure

- `./notebooks/` - Jupyter Notebooks
- `./reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `./tools/` - any 3rd party tools and scripts
- `./htm_rl/htm_rl/` - package src code
  - `run_mdp_test.py` - entry point with usage example

## Quick intro

### HTM

Ordered list of useful links to dive into the HTM theory and practice:

- [HTM School](https://www.youtube.com/watch?v=XMB0ri4qgwc&list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9) youtube playlist
  - 0-12 are required, others - optional
  - important to get used to: SDR concept and feature; concept and high level work of Spatial Pooler (SP) and Temporal Memory (TM)
  - NB: implementation details aren't important at first, no need for grocking them from the start
- [Optional] Short and easy to read [numenta paper](https://arxiv.org/abs/1503.07469) on the power of SDR.
- Intro to `htm.core` package Temporal Memory class - two-part blog post series ([part1](https://3rdman.de/2020/02/hierarchical-temporal-memory-part-1-getting-started/), [part2](https://3rdman.de/2020/04/hierarchical-temporal-memory-part-2/)) on how to use it.
  - 01-03 notebooks in `./notebooks/` are complementary, check out them.
  - NB: it's good if you understood high level TM algo and at this point have questions regarding the low level details
- [BAMI.TM](https://numenta.com/assets/pdf/temporal-memory-algorithm/Temporal-Memory-Algorithm-Details.pdf) - definitive guide on Temporal Memory (TM) implementation details
  - very important for understanding HTM - how to use it and its limits
  - 04-05 notebooks in `./notebooks/` are complementary, check out them.

Additional useful links:

- [BAMI.SDR](https://numenta.com/assets/pdf/biological-and-machine-intelligence/BaMI-SDR.pdf) - definitive guide on Sparse Distributed Representations (SDR)
- [BAMI.SP](https://numenta.com/assets/pdf/spatial-pooling-algorithm/Spatial-Pooling-Algorithm-Details.pdf) - definitive guide on Spatial Pooler (SP)
- [Optional] Another [numenta paper](https://arxiv.org/abs/1903.11257) on the power of sparse representations applied to mainstream Artificial Neural Networks.
