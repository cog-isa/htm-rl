import random

from htm_rl.scenarios.config import FileConfig
from htm_rl.scenarios.experiment import Experiment
from htm_rl.scenarios.standard.run_stats import RunStats
from htm_rl.scenarios.sweep.scenario import Scenario
from htm_rl.scenarios.utils import add_overwrite_attributes, ProgressPoint
from htm_rl.scenarios.wandb_utils import init_wandb_run


class SweepExperiment(Experiment):
    pre_subconfig_reading_prefix = 'FF0.'
    config: FileConfig

    def __init__(self, config: FileConfig):
        self.config = config

    def run(self,):
        config = self.config

        add_overwrite_attributes(config, config['overwrites'], prefix=self.pre_subconfig_reading_prefix)
        config.read_subconfigs('envs', prefix='env')
        config.read_subconfigs('agents', prefix='agent')
        for key, agent in config['agents'].items():
            agent.read_subconfig('sa_encoder', 'sa_encoder')
        add_overwrite_attributes(config, config['overwrites'], ignore=self.pre_subconfig_reading_prefix)

        self._try_append_wandb_group_id()
        self.config.autosave()

        env_seeds = config['env_seeds']
        agent_seeds = config['agent_seeds']

        wandb_run = init_wandb_run(self.config)
        experiment_progress = ProgressPoint()
        total_steps = 0

        for env_seed in env_seeds:
            config['env_seed'] = env_seed
            agent_results = []
            for agent_seed in agent_seeds:
                # FIX ME
                # agent_seed = random.randint(0, 1000000)
                config['agent_seed'] = agent_seed
                print(f'AGENT: {config["agent"]}     SEED: {env_seed} {agent_seed}')

                scenario = Scenario(
                    config, progress=experiment_progress, total_steps=total_steps,
                    **config
                )
                train_results = scenario.run(wandb_run)
                train_results.print_results()
                agent_results.append(train_results)
                total_steps = scenario.total_steps

            if len(agent_seeds) > 1:
                results = RunStats.aggregate_stats(agent_results)
                results.print_results()
            print('')

    def _try_append_wandb_group_id(self):
        if not self.config['wandb.enabled']:
            return
        from wandb.util import generate_id
        self.config['wandb.group'] = generate_id()
