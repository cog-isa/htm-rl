# How to run HIMA

Sign up to [wandb](https://wandb.ai/) and get access token in your profile settings.

## Run one experiment

Run

``` bash
# replace <config name> with the config filename from
# `../../experiments/htm_agent/configs` without extension
python htm_agent.py <config name>
```

Do not forget to change `entity` parameter in corresponding config file to match your [wandb](https://wandb.ai/) login. When wandb asks you to login for the first time, use your access token obtained earlier.

## Run Sweep

Wandb [sweep](https://docs.wandb.ai/guides/sweeps) runs series of experiments with different seeds and parameters:

```bash
cd ../../experiments/htm_agent

# replace <sweep config name> with the sweep config filename without extension
wandb sweep sweep/<sweep config name>

# replace <sweep id> with the returned id
python scripts/run_agents.py -n n_processes -c "wandb agent <sweep id>"
```
