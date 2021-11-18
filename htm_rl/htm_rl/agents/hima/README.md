## How to run HIMA

0. Sign up to [wandb](https://wandb.ai/).
1. Install [htm.core](https://github.com/ZhekaHauska/htm.core) from source.
2. Install `htm_rl`(this repo) and requirements.

### Run one experiment:

3. Run `python htm_agent.py config_name`,
where config name should be from `ls htm_rl/htm_rl/experiments/htm_agent/configs`
without extension. Don't forget to change `entity` parameter
in corresponding config file to match your [wandb](https://wandb.ai/) login.

### Run Sweep

[Sweep](https://docs.wandb.ai/guides/sweeps) is a series of experiments with different seeds and parameters.

3. `wandb sweep path_to_config`, default sweep configs are in `htm_rl/htm_rl/experiments/htm_agent/sweeps`.
4. `python htm_rl/htm_rl/experiments/htm_agent/scripts/run_agents.py -n n_processes -c "wandb agent sweep_id"`