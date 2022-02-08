# HIMA: Hierarchical Intrinsically Motivated Agent Planning Behavior with Dreaming in Grid Environments

- [HIMA: Hierarchical Intrinsically Motivated Agent Planning Behavior with Dreaming in Grid Environments](#hima-hierarchical-intrinsically-motivated-agent-planning-behavior-with-dreaming-in-grid-environments)
  - [Links](#links)
  - [Quick install](#quick-install)
  - [Repository structure](#repository-structure)
  - [Run examples](#run-examples)
    - [How to run HIMA agent](#how-to-run-hima-agent)
      - [Run one experiment](#run-one-experiment)
      - [Run Sweep](#run-sweep)
    - [Non-HIMA Q-learning agent](#non-hima-q-learning-agent)

In this repository, we present a model of the autonomous agent called HIMA (Hierarchical Intrinsically Motivated Agent). Its modular structure is divided into blocks that consist of unified and reusable sub-blocks. We provide HIMA with a novel hierarchical memory model. Its Spatial Pooler and Temporal Memory sub-blocks are based on corresponding objects from the Hierarchical Temporal Memory model. However, we contribute to it by extending Temporal Memory with external modulation support via feedback connections and the higher-order sequences learning algorithm. The latter enables us to construct a hierarchy that can work with the state and action abstractions. We also propose the Basal Ganglia model and empowerment as two further building sub-blocks, which are responsible for learning the action selection strategy while being driven by the modulated motivation signal. Additionally, we supply our agent with an ability to learn in imagination that we call dreaming. The sparse distributed representation of states and actions is another distinguishing feature of our model. As a result, our contribution is to investigate the representation of abstract context-dependent actions that denote behavioral programs, as well as the ability of the basal ganglia to learn the choosing strategy between partially overlapping actions. Finally, we validate HIMA's ability to aggregate and reuse experience in order to solve RL tasks with changing goals.

## Links

- This repo [link](https://github.com/cog-isa/htm-rl)
- Contributors [guide](./CONTRIBUTING.md)
- Cumulative project's [readme](./htm_rl/htm_rl/README.md)
- Introductory [materials](./intro.md)

## Quick install

There're two setup guides:

- [quick & short version](#quick-install) is here. It's recommended for `htm_rl` library users.
- [extended version](./install.md/#install-requirements) is for contributors or if you have troubles with this version.

Before cloning the repository, make sure Git LFS is installed (see [help](./install.md/#git-lfs)). Then:

```bash
# create env with required packages via conda, then activate it
conda create --name hima python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock imageio seaborn
conda activate hima

# install packages, that cannot be installed with conda, with pip
pip install hexy prettytable "pytest>=4.6.5"

# git clone our `htm.core` fork to an arbitrary place and pip install it from sources
# pip will install missing dependencies to the current environment if needed
cd <where to clone>
git clone https://github.com/ZhekaHauska/htm.core.git
cd htm.core
pip install --use-feature=in-tree-build .

#  cd to the htm_rl subdirectory in project root and install htm_rl package
cd <hima_project_root>/htm_rl
pip install -e .
```

## Repository structure

- `notebooks/` - Jupyter Notebooks
- `reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `tools/` - any 3rd party tools and scripts
- `watcher/` - visualization tool for HTM SP and TM.
- `htm_rl/` - sources root (for PyCharm), it contains `setup.py`
  - `htm_rl/` - `htm_rl` package sources

## Run examples

### How to run HIMA agent

Sign up to [wandb](https://wandb.ai/) and get access token in your profile settings.

#### Run one experiment

``` bash
# cd to package sources root
cd <hima_project_root>/htm_rl/htm_rl

# replace <config name> with the config filename without extension
python agents/htm/htm_agent.py experiments/htm_agent/configs/<config_name>
```

Do not forget to change `entity` parameter in corresponding config file to match your [wandb](https://wandb.ai/) login. When wandb asks you to login for the first time, use your access token obtained earlier.

#### Run Sweep

Wandb [sweep](https://docs.wandb.ai/guides/sweeps) runs series of experiments with different seeds and parameters.

```bash
# cd to package sources root
cd <hima_project_root>/htm_rl/htm_rl

cd experiments/htm_agent

# replace <sweep config name> with the sweep config filename without extension
wandb sweep sweep/<sweep config name>

# replace <sweep id> with the returned id
python scripts/run_agents.py -n n_processes -c "wandb agent <sweep id>"
```

### Non-HIMA Q-learning agent

```bash
# cd to package sources root
cd <hima_project_root>/htm_rl/htm_rl/

# cd to the 5x5_pos experiments
cd experiments/5x5_pos/

# runs random agent and Q-learning agent with learned model 
# on 5x5 env with an agent position as the observation
python ../../run_experiment.py -c debug -e pos -a rnd qmb
```
