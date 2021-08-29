# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Installation](#installation)
    - [[Optional] Create dedicated python environment](#optional-create-dedicated-python-environment)
    - [[Optional] Install conda](#optional-install-conda)
    - [Requirements.txt structure](#requirementstxt-structure)
    - [Install requirements to dedicated python environment](#install-requirements-to-dedicated-python-environment)
    - [Install `htm.core`](#install-htmcore)
    - [[Optional] Install `htm_rl` package](#optional-install-htm_rl-package)
  - [Repository structure](#repository-structure)
  - [Quick intro](#quick-intro)
    - [HTM](#htm)

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

### [Optional] Create dedicated python environment

Create new environment with _conda_ or any alternative like _pipenv_ or _virtualenv_. The name of the environment is up to you, we use `htm` for this guide.
  
  ```bash
  conda create --name htm
  conda activate htm
  ```

### [Optional] Install conda

Using conda as environment/package manager is not necessary - pip + env managers like _pipenv_ or _virtualenv_ are fine too. Although, we recommend to prioritize using conda over pip, because it allows managing not only python packages as pip does and also resolves package dependencies smarter, ensuring that the environment has no version conflicts (see [understanding pip and conda](https://www.anaconda.com/blog/understanding-conda-and-pip) for details). So, the recommended option is to install as much packages as you can with conda, and use pip only as the last resort (sadly, not everything is available as conda packages).

If you are new to conda, we recommend to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (a minimal console version of conda) instead of Anaconda (see [the difference](https://stackoverflow.com/a/45421527/1094048)).

If you brave enough, as alternative to Miniconda, you may also consider installing [Miniforge](https://github.com/conda-forge/miniforge) - the difference from Miniconda is only that it sets default conda package channel to [conda-forge](https://github.com/conda-forge/miniforge), which is community-driven so it gets updates faster in general and contains more packages. Another alternative is to use [Mamba](https://github.com/mamba-org/mamba) instead of conda - Mamba mimics conda API, even uses it and fully backward compatible with it, so you can use both together; Mamba has enhanced dependency-resolving performance and async package installation, and also extends conda functionality. Last two options combined, [Mamba + Miniforge = Mambaforge](https://github.com/conda-forge/miniforge#mambaforge), are distributed by Miniforge community.

### Requirements.txt structure

File `requirements.txt` contains all required dependencies. Dependencies are grouped by what you can install with conda or pip and what is exclusive to pip.

Sadly, even if you are going to use pip exclusively, it's unlikely that you will manage to install the requirements with the usual `pip install -r requirements.txt` as at the time of writing (Sep 2021), `htm.core` pip package is broken and has to be installed manually from sources (see below).

### Install requirements to dedicated python environment

Create a new environment with the packages from conda section ("conda requirements") in `requirements.txt`, then activate this environment. The name of the environment is up to you, we use `htm` for this guide. Please, check that the package list below matches `requirements.txt` before running the command below:

```bash
conda create --name htm python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock
conda activate htm
```

Install packages from pip requirements group in `requirements.txt` with pip except `htm.core`. Again, check that the package list below matches `requirements.txt`:

```bash
pip install hexy prettytable pytest>=4.6.5
```

### Install `htm.core`

From sources (the only working option, Sep 2021):

```bash
git clone https://github.com/htm-community/htm.core
cd htm.core
pip install --use-feature=in-tree-build .
```

From test-PyPi (broken, Sep 2021):

```bash
pip install -i https://test.pypi.org/simple/ htm.core
```

### [Optional] Install `htm_rl` package

To have an ability to import and use `htm_rl` package outside of the package root folder, e.g. in Jupyter Notebooks or in another packages, install it with _development mode_ flag:

```bash
cd <htm_rl_project_root>/htm_rl
pip install -e .
```

Now you can import modules from the `htm_rl` package.

## Repository structure

- `notebooks/` - Jupyter Notebooks
- `reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `tools/` - any 3rd party tools and scripts
- `watcher/` - visualization tool for HTM SP and TM.
- `htm_rl/` - sources root (mark this directory in for PyCharm), it contains `setup.py`
  - `htm_rl/` - `htm_rl` package sources
    - `run_X.py` - runners, i.e. entry point to run testing scenarios

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
