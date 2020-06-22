# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Installation](#installation)
    - [Python package requirements](#python-package-requirements)
    - [Jupyter Notebooks stripping](#jupyter-notebooks-stripping)
    - [htm_rl package use in Jupyter Notebooks](#htm_rl-package-use-in-jupyter-notebooks)
  - [Project structure and guideline](#project-structure-and-guideline)
  - [Quick intro](#quick-intro)
    - [HTM](#htm)
    - [htm_rl package](#htm_rl-package)
  - [TODO](#todo)
  - [Thoughts](#thoughts)

## Links

- This repo [link](https://github.com/cog-isa/htm-rl)
- Cumulative [FAQ](htm_rl/htm_rl/README.md)

## Installation

### Python package requirements

`[Optional]` Create environment `htm_rl` with _conda_ (alternatives are _pipenv_ and _virtualenv_):
  
```bash
conda create --name htm_rl
conda activate htm_rl
```

Install requirements [specified in _requirements.txt_]:

```bash
pip install -r requirements.txt
```

### Jupyter Notebooks stripping

Most of the time it's useful to cut metadata and output from Jupyter Notebooks. Hence we use git filters as mentioned [here](https://stackoverflow.com/a/60859698/1094048). All sctipts are taken from [fastai-nbstripout](https://github.com/fastai/fastai-nbstripout).

To set everything up just run:

```bash
tools/run-after-git-clone
```

As a result, folders dedicated for notebooks will have

- `./notebooks/`: output erased
- `./reports/`: output retained

### htm_rl package use in Jupyter Notebooks

To have an ability to import and use `htm_rl` package in Jupyter Notebooks, install it with _development mode_ flag:

```bash
cd <project_root>/htm_rl

pip install -e .
```

## Project structure and guideline

Project structure:

- `./notebooks/` - Jupyter Notebooks
- `./reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `./tools/` - any 3rd party tools and scripts
- `./htm_rl/htm_rl/` - `htm_rl` package src code
- __NB__: these files are added to _.gitignore_ so you can use them as local temporal scratchpad
  - `./notebooks/00_scratchpad.ipynb`
  - `./htm_rl/htm_rl/scratch.py`

## Quick intro

### HTM

Ordered list of useful links aka _guide_:

- [HTM School](https://www.youtube.com/watch?v=XMB0ri4qgwc&list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9) youtube playlist
  - 0-12 are required, others - optional
  - important: SDR concept and feature; concept and high level work of Spatial Pooler (SP) and Temporal Memory (TM)
  - NB: implementation details aren't important at first, no need for grocking them right now
- [Optional] Short and easy to read [numenta paper](https://arxiv.org/abs/1503.07469) on the power of SDR.
- Intro to _htm.core_ Temporal Memory class - two-part blog post series ([part1](https://3rdman.de/2020/02/hierarchical-temporal-memory-part-1-getting-started/), [part2](https://3rdman.de/2020/04/hierarchical-temporal-memory-part-2/)) on how to use it.
  - 01-03 notebooks in `./notebooks/` are complementary, check out them.
  - NB: it's good if you got understanding of high level TM algo and now have questions regarding the low level details
- [BAMI.TM](https://numenta.com/assets/pdf/temporal-memory-algorithm/Temporal-Memory-Algorithm-Details.pdf) - definitive guide on Temporal Memory (TM) implementation details
  - very important for understanding HTM - how to use it and its limits
  - 04-05 notebooks in `./notebooks/` are complementary, check out them.

Additional useful links:

- [BAMI.SDR](https://numenta.com/assets/pdf/biological-and-machine-intelligence/BaMI-SDR.pdf) - definitive guide on Sparse Distributed Representations (SDR)
- [BAMI.SP](https://numenta.com/assets/pdf/spatial-pooling-algorithm/Spatial-Pooling-Algorithm-Details.pdf) - definitive guide on Spatial Pooler (SP)
- [Optional] Another [numenta paper](https://arxiv.org/abs/1903.11257) on the power of sparse representations applied to mainstream Artificial Neural Networks.

### htm_rl package

*Outdated*

Entry points are `run_mdp_test.py` and `run_gridworld_test.py`. The former is for simpler develop/debug purposes as you can test implementation on a simple low-dimentional MDP. The latter is for testing/evaluation purposes as you have more complex environrment (taken from gym-minigrid package).

## TODO

- [x] Adapt planning algo for high order TMs
- [x] Make semi-handcrafted POMDP version of gridworld generator
  - naive version w/ ray casting maybe is enough
- [ ] config based tests
  - [ ] fine grained trace verbosity levels
  - [ ] test config + builder classes
- [ ] make FAQ on TM params
- [ ] issue: fantom false positive transitions
  - e.g. sometimes it predicts state superposition from sar - it's a mistake as the next state is always _one_ determined state.
- [ ] start testing on POMDPs

## Thoughts

- consider using SP between an input an TM
  - only states need SP, because actions and reward are just ints (naive encoding is enough)
  - concat them together
  - it will take care of sparsity
  - maybe smoothes the difference in size for a range of diff environments
    - bc even large envs may have a very small signal
- consider TD($\lambda$)-based approach from Sungur's work
- split SAR-based TM into State TM + Action TM
  - both has apical connections to each other
  - reward or goal-based approach? Could be interchangeable
- goal-based hierarchies of TM blocks
- SP hierarchies for large input images
  - with every SP working similar to convolution filter
- consider doing live-logging experiments in markdown there
