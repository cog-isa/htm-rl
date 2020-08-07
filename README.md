# HTM applied to RL setting

- [HTM applied to RL setting](#htm-applied-to-rl-setting)
  - [Links](#links)
  - [Installation](#installation)
    - [Python package requirements](#python-package-requirements)
    - [Jupyter Notebooks stripping](#jupyter-notebooks-stripping)
      - [Jupyter Notebooks trusting and stripping setup details](#jupyter-notebooks-trusting-and-stripping-setup-details)
    - [Using `htm_rl` package in Jupyter Notebooks](#using-htm_rl-package-in-jupyter-notebooks)
    - [Git LFS: working with data files](#git-lfs-working-with-data-files)
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

1. `[Optional]` Create environment `htm_rl` with _conda_ (alternatives are _pipenv_ and _virtualenv_):
  
    ```bash
    conda create --name htm_rl
    conda activate htm_rl
    ```

2. Install requirements [specified in _requirements.txt_]:

    ```bash
    pip install -r requirements.txt
    ```

### Jupyter Notebooks stripping

Most of the time it's useful to cut metadata and output from Jupyter Notebooks. Hence we use git filters as mentioned [here](https://stackoverflow.com/a/60859698/1094048). All sctipts are taken from [fastai-nbstripout](https://github.com/fastai/fastai-nbstripout).

To set everything up just run:

```bash
tools/run-after-git-clone
```

Folders dedicated for notebooks are:

- `./notebooks/`: output erased
- `./notebooks/reports/`: output retained
  - so, it's a special subfolder in `notebooks` for this purpose!

Also notebooks can be included in reports, hence it has stripping [and trusting] rule too:

- `./reports/`: output retained

#### Jupyter Notebooks trusting and stripping setup details

_Skip this section if you just want to install the repo._

This section consists of details about how it was done, in case it has to be re-done again. Rationale:

- stripping is for stripping unneccessary metadata
- trusting
  - normally, modified outside of jupyter environment notebooks lose their "trusted" state, so you won't be able to run them automatically and will need to manually set them to be trusted.
  - the following setup does it automatically for you on `git pull`

Steps:

- Configure which directories you want to be "auto-trusted" by editing `tools/trust-doc-nbs`:

  ```python
  trust_nbs('./notebooks')
  trust_nbs('./reports')
  ```

  - if you ever need to reconfigure the set of trusted folders, just edit the file again and nothing more :)
- Add `.gitconfig` to `.gitignore`
- Configure which directories you want to strip [and how] by placing the right `.gitattributes` files into these folders.
  - This `.gitattributes` will strip out all the unnecessary bits and keep the `output`s:

    ``` python
    *.ipynb filter=fastai-nbstripout-code
    *.ipynb diff=ipynb-code
    ```

  - This `.gitattributes` will strip out all the unnecessary bits, including the `output`s:

    ```python
    *.ipynb filter=fastai-nbstripout-docs
    *.ipynb diff=ipynb-docs
    ```

  - If you ever need to change which and how directories have to be stripped, just operate with `.gitattributes` files, i.e. nothing has to be done with git or `tools/` scripts. But remember about the set of trusting folders too.

- `git add` created `.gitattributes` files before proceeding
  - as I understand, you don't have to commit them
- run `tools/run-after-git-clone` to setup things, which
  - auto-generates `.gitconfig`
  - sets up git hooks

After it's done and commited only the last step is have to be done by any other repo users on git clone [exactly as it's said in the guide in previous section].

### Using `htm_rl` package in Jupyter Notebooks

To have an ability to import and use `htm_rl` package in Jupyter Notebooks, install it with _development mode_ flag:

```bash
cd <project_root>/htm_rl

pip install -e .
```

### Git LFS: working with data files

TL;DR for Linux:

```bash
apt-get install git-lfs && git lfs install
```

___

Storing binary or any other "data" files in a repository can quickly bloat its size. They rarely change, and if so, mostly as a whole (i.e they are added or deleted, not edited). But git still treats them as text files and tries to track their content, putting it to its index.

If only was a way to track some signature of these files, store their content somewhere else and pull them only on demand - i.e. on checkout!

Fortunately, that's exactly what Git[hub] LFS, which stands for Large File Storage, do :)

> Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

To setup Git LFS you need to [download and install](https://github.com/git-lfs/git-lfs#downloading) `git-lfs` first. E.g. for Linux it's just a:

```bash
apt-get install git-lfs
```

Then "link" git and git-lfs together by:

```bash
git lfs install
```

After that you just work as usual :)

## Project structure and guideline

Project structure:

- `./notebooks/` - Jupyter Notebooks
- `./reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `./tools/` - any 3rd party tools and scripts
- `./htm_rl/htm_rl/` - `htm_rl` package src code
  - `./experiments/` - configs and results of experiments
- __NB__: the following files are added to _.gitignore_ so you can use them as a local temporal scratchpad
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

Entry point is `run_mdp_test.py`.

## TODO

**Urgent**:

- [ ] add ruamel.yaml to requirements
- [ ] describe config based building details
  - conventions
  - implementation
  - patches

Research + functional tasks

- [ ] Adapt planning to goal-based strategy
  - [x] Switch from reward-based planning to goal-based
    - [x] Cut out reward from encoding and memorizing
    - [x] Track history of rewarding states and plan according to any of them
      - add naive list-based rewarding states tracking
  - [ ] Test transfer learning capabilities
    - [ ] Adapt environments for random initial states
    - [ ] Adapt environments for random rewarding states
    - [x] Make the set of testing environments
    - [ ] Adapt test runners for a multi-environment tests
    - [x] Make config for an experiment
    - [x] Run experiment
  - [ ] Report results
    - [ ] Update method description
    - [ ] Add experiment results
- Not acknowledged and questionable:
  - [ ] Split SAR TM into 2 TMs
    - State TM: (s, a) $\rightarrow$ s'
    - Action TM: s $\rightarrow$ a
    - Direct external rewards aren't a thing
    - Reinforcement isn't tracked ATM
  - [ ] Investigate `MaxSegmentsPerCell` parameter impact
  - [ ] Implement integer encoder w/ overlapping buckets
    - overlapping should be a parameter
    - it defines the level of uncertainty
    - MDP planning becomes a light version of POMDP planning because of uncertainty
  - [ ] Investigate relation between overlapping and sufficient activation thresholds
  - [ ] Investigate `MaxSynapsesPerSegment` parameter impact
  - [ ] Start testing on POMDPs

Non-critical issues needing further investigation

Auxialiary tasks, usability improvements and so on

- [x] make FAQ on TM params
- [x] config based tests
  - [x] test config + builder classes
  - [x] improve config based building:
    - one config file for one test run (=all agents one test)
    - or even one config file for the whole experiment (=all agents all tests)
- [x] fine grained trace verbosity levels
- [x] setup release-based dev cycle
  - add tagging to git commits
  - how to add release notes
  - ?notes for major releases should contain algo details from FAQ
- [x] release v0.1 version of the SAR-based agent

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
