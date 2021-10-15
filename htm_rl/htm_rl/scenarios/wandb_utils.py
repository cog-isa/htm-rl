def read_wandb_config(config):
    wandb_config: dict = config['wandb']

    keys = ['project', 'entity', 'sweep_id'] #, 'group']
    for key in keys:
        if key in config and config[key] is not None:
            wandb_config[key] = config[key]

    wandb_init_kwargs = {
        key: wandb_config[key]
        for key in keys
        if key in wandb_config
    }
    wandb_config['init_kwargs'] = wandb_init_kwargs

    # if 'group' in wandb_config['init_kwargs']:
    #     wandb_init_kwargs['job_type'] = f'eval'

    return wandb_config


def init_wandb_run(config):
    wandb_config = read_wandb_config(config)
    if not wandb_config['enabled']:
        return None

    import wandb
    assert wandb_config['project'] is not None, \
        'Wandb project name, set by `wandb.project` config field, is missing.'

    from htm_rl.scenarios.config import FileConfig
    config = config.as_dict() if isinstance(config, FileConfig) else config

    run = wandb.init(
        # reinit=True,
        dir=config['base_dir'], config=config,
        **wandb_config['init_kwargs']
    )
    return run