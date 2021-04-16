# Development setup

- [Development setup](#development-setup)
  - [Python package requirements](#python-package-requirements)
  - [Working with Jupyter Notebooks](#working-with-jupyter-notebooks)
    - [Using `htm_rl` package in Jupyter Notebooks](#using-htm_rl-package-in-jupyter-notebooks)
    - [Jupyter Notebooks stripping](#jupyter-notebooks-stripping)
  - [Working with data files](#working-with-data-files)
    - [Git LFS](#git-lfs)
  - [Project structure](#project-structure)
  - [What's next](#whats-next)
  - [Additional notes](#additional-notes)
    - [Jupyter Notebooks trusting and stripping setup details](#jupyter-notebooks-trusting-and-stripping-setup-details)

This guide mostly aims the developers who wants to join us and contribute to the project.

It also covers some additional topics regarding development setup in order to easily reproduce them and rationale behind it.

## Python package requirements

1. `[Optional]` Create new environment (e.g. *htm_rl*) with _conda_ or any alternative like _pipenv_ or _virtualenv_:
  
    ```bash
    conda create --name htm_rl
    conda activate htm_rl
    ```

2. Install requirements specified in _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

We recommend prioritize using conda as a package and environments manager, because it allows managing not only python packages as pip does. Also it resolves package dependencies smarter, ensuring that the environment has no version conflicts.

**NB**: So in the second step you could try to install as much packages as you can with conda and only after that use pip as a last resort. Unfortunately, at the moment of writing this guide this way gives a lot more pain :(

## Working with Jupyter Notebooks

### Using `htm_rl` package in Jupyter Notebooks

To have an ability to import and use `htm_rl` package in Jupyter Notebooks, install it with _development mode_ `-e` flag:

```bash
cd <project_root>/htm_rl

pip install -e .
```

### Jupyter Notebooks stripping

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

## Working with data files

Storing binary or any other "data" files in a repository can quickly bloat its size. They rarely change, and if so, mostly as a whole file (i.e they are added or deleted, not edited). But git still treats them as text files and tries to track their content, putting it to the index.

If only was a way to track some signature of these files, store their content somewhere else and pull them only on demand - i.e. on checkout!

Fortunately, that's exactly what Git[hub] LFS, which stands for Large File Storage, do :)

> Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise.

### Git LFS

To setup Git LFS you need to [download and install](https://github.com/git-lfs/git-lfs#downloading) `git-lfs` first. As an example for Linux it's just a:

```bash
apt-get install git-lfs
```

Then "link" git and git-lfs together by:

```bash
git lfs install
```

After that you just work as usual :)

## Project structure

Project structure:

- `./notebooks/` - Jupyter Notebooks
- `./reports/` - any [markdown, tex, Jupyter Notebooks] reports
- `./tools/` - any 3rd party tools and scripts
- `./htm_rl/htm_rl/` - package src code
  - `./experiments/` - configs and results of experiments
  - `run_mdp_test.py` - common entry point with usage example
- __NB__: the following files are added to _.gitignore_ so you can use them as a local temporal scratchpad
  - `./notebooks/00_scratchpad.ipynb`
  - `./htm_rl/htm_rl/scratch.py`

## What's next

If you're new to the HTM, check out [this](./README.md#quick-intro) quick intro list. After that proceed to the project's [readme](htm_rl/htm_rl/README.md).

## Additional notes

### Jupyter Notebooks trusting and stripping setup details

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
