from itertools import product

from htm_rl.scenarios.config import FileConfig
from htm_rl.scenarios.experiment import Experiment
from htm_rl.scenarios.standard.scenario import RunStats
from htm_rl.scenarios.dream_cond.scenario import Scenario
from htm_rl.scenarios.utils import add_overwrite_attributes, get_filtered_names_for


class DreamingConditionExperiment(Experiment):
    config: FileConfig

    def __init__(self, config: FileConfig):
        self.config = config

    def run(self,):
        config = self.config

        config.read_subconfigs('envs', prefix='env')
        config.read_subconfigs('agents', prefix='agent')
        for key, agent in config['agents'].items():
            agent.read_subconfig('sa_encoder', 'sa_encoder')
        add_overwrite_attributes(config, config['overwrites'])

        env_seeds = config['env_seeds']
        agent_seeds = config['agent_seeds']
        env_names = get_filtered_names_for(config, 'envs')
        agent_names = get_filtered_names_for(config, 'agents')

        experiment_setups = list(product(env_names, agent_names, env_seeds))
        for env_name, agent_name, env_seed in experiment_setups:
            config['env'] = env_name
            config['env_seed'] = env_seed
            config['agent'] = agent_name
            agent_results = []
            for agent_seed in agent_seeds:
                config['agent_seed'] = agent_seed
                print(f'AGENT: {agent_name}     SEED: {env_seed} {agent_seed}')

                results: RunStats = Scenario(config).run()
                results.print_results()
                agent_results.append(results)

            if len(agent_seeds) > 1:
                results = RunStats.aggregate_stats(agent_results)
                results.print_results()
            print('')
