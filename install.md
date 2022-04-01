# Installation

- [Installation](#installation)
  - [Install requirements](#install-requirements)
    - [[Optional] Install conda](#optional-install-conda)
    - [[Info] requirements.txt structure](#info-requirementstxt-structure)
    - [Install requirements to dedicated python environment](#install-requirements-to-dedicated-python-environment)
      - [Option 1: use conda and pip](#option-1-use-conda-and-pip)
      - [Option 2: use pip only](#option-2-use-pip-only)
    - [Install `htm.core`](#install-htmcore)
    - [[Optional] Install `htm_rl` package](#optional-install-htm_rl-package)
  - [Contributors setup](#contributors-setup)
    - [Working with Jupyter Notebooks](#working-with-jupyter-notebooks)
      - [Using `htm_rl` package in Jupyter Notebooks](#using-htm_rl-package-in-jupyter-notebooks)
      - [Jupyter Notebooks stripping](#jupyter-notebooks-stripping)
    - [Working with data files](#working-with-data-files)
      - [Git LFS](#git-lfs)
    - [Project structure](#project-structure)
    - [What's next](#whats-next)
    - [Additional notes](#additional-notes)
      - [Jupyter Notebooks trusting and stripping setup details](#jupyter-notebooks-trusting-and-stripping-setup-details)

This guide mostly aimed for contributors of our project. Besides general [Install requirements](#install-requirements) section, in [Contributors setup](#contributors-setup) it covers some additional topics regarding developers setup with the rationale behind it.

## Install requirements

### [Optional] Install conda

Using conda as an environment and package manager is optional. Using pip as a package manager and  _pipenv_/_virtualenv_/etc. as Python environment managers is a valid option too.

However, we recommend to prioritize using conda over pip, because it allows managing not only python packages, as pip does, and also resolves package dependencies smarter, ensuring that the environment has no version conflicts (see [understanding pip and conda](https://www.anaconda.com/blog/understanding-conda-and-pip) for details). Having said that, we recommend trying to install as much packages as you can with conda first, and use pip only as the last resort (sadly, not everything is available in conda channels).

If you are new to conda, we recommend getting conda with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (a minimal console version of conda) instead of Anaconda (see [the difference](https://stackoverflow.com/a/45421527/1094048)).

*If you brave enough, you may also consider [Miniforge](https://github.com/conda-forge/miniforge) as alternative to Miniconda - the difference from Miniconda is only that it sets default conda package channel to [conda-forge](https://github.com/conda-forge/miniforge), which is community-driven so it gets updates faster in general and contains more packages. Another alternative is to use [Mamba](https://github.com/mamba-org/mamba) instead of conda - Mamba mimics conda API, even uses it and fully backward compatible with it, so you can use both together; Mamba has enhanced dependency-resolving performance and async package installation, and also extends conda functionality. Last two options combined, [Mamba + Miniforge = Mambaforge](https://github.com/conda-forge/miniforge#mambaforge), are distributed by Miniforge community.*

### [Info] requirements.txt structure

File `requirements.txt` contains all required dependencies. Dependencies are grouped by what you can install with conda or pip (conda section) and what is exclusive to pip (pip section). Besides what mentioned in `requirements.txt` you will also need to install our fork of the `htm.core` package from sources. In following sections we provide an installation guide.

### Install requirements to dedicated python environment

In this section we provide you with two options: install requirements using both conda and pip or install everything with pip. In both cases requirements are installed into the new conda environment. The name of the environment is up to you, we use `htm` for this guide.

*Do not hesitate to check that the package lists in the commands below match requirements in `requirements.txt`.*

#### Option 1: use conda and pip

Create new conda environment and install the conda requirements group, then install pip requirements group:

```bash
conda create --name htm python=3.9 numpy matplotlib jupyterlab ruamel.yaml tqdm wandb mock imageio seaborn
conda activate htm
pip install hexy prettytable pytest>=4.6.5
```

#### Option 2: use pip only

Create new conda environment and install everything with pip:

```bash
conda create --name htm python=3.9
conda activate htm
pip install -r requirements.txt
```

### Install `htm.core`

In our research we use slightly modified version of the `htm.core` package, so you have to install our fork instead of the origin. The following script installs it from the sources:

```bash
cd <where to fork>
git clone https://github.com/ZhekaHauska/htm.core.git
cd htm.core
pip install --use-feature=in-tree-build .
```

### [Optional] Install `htm_rl` package

Install our `htm_rl` package:

```bash
cd <htm_rl_project_root>/hima
pip install -e .
```

The last command installs `htm_rl` with _development mode_ flag, which allows you importing modules from the `htm_rl` package outside of the package root folder, e.g. in Jupyter Notebooks or in another packages.

Development mode flag `-e` prevent package sources to be copied to the index. Instead, the symlink is created, which means that the edits are "visible" immediately and don't require you to reinstall/update package manually after any changes in the package sources.

## Contributors setup

### Working with Jupyter Notebooks

#### Using `htm_rl` package in Jupyter Notebooks

See [[Optional] Install `htm_rl` package](#optional-install-htm_rl-package).

#### Jupyter Notebooks stripping

Most of the time it's useful to cut metadata and output from Jupyter Notebooks. Hence we use git filters as mentioned [here](https://stackoverflow.com/a/60859698/1094048). All sctipts are taken from [fastai-nbstripout](https://github.com/fastai/fastai-nbstripout).

To set everything up run:

```bash
cd <project root folder>
./tools/run-after-git-clone
```

Folders dedicated for notebooks are:

- `./notebooks/`: output erased
- `./notebooks/reports/`: output retained
  - so, it's a special subfolder in `notebooks` for this purpose!

Also notebooks can be included in reports, hence the folder dedicated for reports has stripping [and trusting] rule too:

- `./reports/`: output retained

### Working with data files

Storing binary or any other "data" files in a repository can quickly bloat its size. They rarely change, and if so, mostly as a whole file (i.e they are added or deleted, not edited). But git still treats them as text files and tries to track their content, putting it to the index.

If only was a way to track some signature of these files, store their content somewhere else and pull them only on demand - i.e. on checkout!

Fortunately, that's exactly what Git[hub] LFS, which stands for Large File Storage, do :)

> Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

#### Git LFS

To setup Git LFS you need to [download and install](https://github.com/git-lfs/git-lfs#downloading) `git-lfs` first. For example, on Linux:

```bash
apt install git-lfs
```

Then "link" git and git-lfs together by:

```bash
git lfs install
```

After that you just work as usual :)

### Project structure

- `notebooks/` - Jupyter Notebooks
- `reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `tools/` - any 3rd party tools and scripts
- `watcher/` - visualization tool for HTM SP and TM.
- `htm_rl/` - sources root (mark this directory for PyCharm), it contains `setup.py`
  - `htm_rl/` - `htm_rl` package sources
    - `run_X.py` - runners, i.e. entry point to run testing scenarios

### What's next

If you're new to the HTM, check out [this](./README.md#quick-intro) quick intro list. After that proceed to the project's [readme](hima/README.md).

### Additional notes

#### Jupyter Notebooks trusting and stripping setup details

This section consists of details about how it was done, in case it has to be re-done again. It's not a part of the common setup process! This setup should be done by the repo admin just once. After that everyone else follow the common installation process.

Rationale:

- stripping is for stripping unneccessary metadata
  - most of the time this metadata just clutters up commits diff
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

After it's done and commited, only the last step is have to be done by any other repo users on git clone, which is exactly as it's said in the developer's installation guide.
