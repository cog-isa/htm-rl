# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Quick install](#quick-install)
  - [Repository structure](#repository-structure)
  - [Working example](#working-example)
  - [Quick intro](#quick-intro)
    - [HTM](#htm)

## Links

- This repo [link](https://github.com/cog-isa/htm-rl)
- Contributors setup [guide](./install.md)
- Cumulative project's [readme](./htm_rl/htm_rl/README.md)
- Project releases [folder](./reports) with version-specific reports.

## Quick install

There're two setup guides:

- [quick&short version](#quick-install) is here. It's recommended for `htm_rl` library users.
- [extended version](./install.md/#install-requirements) is for contributors or if you have troubles with this version.

```bash
# create env with required packages via conda, then activate it
conda create --name htm python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock
conda activate htm

# install packages, that cannot be installed with conda, with pip
pip install hexy prettytable pytest>=4.6.5

# clone htm.core sources to an arbitrary place and pip install it from sources
# pip will install missing htm.core dependencies if needed
git clone https://github.com/htm-community/htm.core
cd htm.core
pip install --use-feature=in-tree-build .

#  cd to the htm_rl subdirectory in project root and install htm_rl package
cd <htm_rl_project_root>/htm_rl
pip install -e .
```

## Repository structure

- `notebooks/` - Jupyter Notebooks
- `reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `tools/` - any 3rd party tools and scripts
- `watcher/` - visualization tool for HTM SP and TM.
- `htm_rl/` - sources root (mark this directory in for PyCharm), it contains `setup.py`
  - `htm_rl/` - `htm_rl` package sources
    - `run_X.py` - runners, i.e. entry point to run testing scenarios

## Working example

```bash
# cd to package sources root
cd <htm_rl_project_root>/htm_rl/htm_rl/

# cd to the 5x5_pos experiments
cd experiments/5x5_pos/

# runs random agent and Q-learning agent with learned model 
# on 5x5 env with an agent position as the observation
python ../../run_experiment.py -c debug -e pos -a rnd qmb
```

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
