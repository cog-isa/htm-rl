# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Quick install](#quick-install)
  - [Repository structure](#repository-structure)
  - [Working example](#working-example)

## Links

- This repo [link](https://github.com/cog-isa/htm-rl)
- Contributors [guide](./CONTRIBUTING.md)
- Cumulative project's [readme](./htm_rl/htm_rl/README.md)
- Introductory [materials](./intro.md)

## Quick install

There're two setup guides:

- [quick & short version](#quick-install) is here. It's recommended for `htm_rl` library users.
- [extended version](./install.md/#install-requirements) is for contributors or if you have troubles with this version.

```bash
# create env with required packages via conda, then activate it
conda create --name htm python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock imageio seaborn
conda activate htm

# install packages, that cannot be installed with conda, with pip
pip install hexy prettytable pytest>=4.6.5

# git clone our `htm.core` fork to an arbitrary place and pip install it from sources
# pip will install missing dependencies to the current environment if needed
cd <where to clone>
git clone https://github.com/ZhekaHauska/htm.core.git
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
