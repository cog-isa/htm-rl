import os.path
import pathlib
from typing import Union

import tqdm
import numpy as np
import imageio
import random
import matplotlib.pyplot as plt
import wandb
from wandb.wandb_run import Run

from htm_rl.agents.v1.agent import RndAgent
from htm_rl.envs.biogwlab.env import BioGwLabEnvironment
from htm_rl.envs.coppelia.environment import ArmEnv


class RndAgentRunner:
    def __init__(self, config: dict, logger: Union[Run, None] = None):

        print('> Start initialization')
        self.rng = np.random.default_rng(config['seed'])
        print('>> Agent is initialising ...')
        self.agent = RndAgent(**config['agent'])
        print('>> Successful')

        print('>> Environment is initialising ...')
        self.env_config = config['environment']
        self.environment_type = config['environment_type']
        if config['environment_type'] == 'gw':
            self.environment = BioGwLabEnvironment(**config['environment'])
        else:
            raise ValueError(f'Unknown environment type: {config["environment_type"]}! ')
        print('>> Successful')

        print('>> Utils are initialising ...')
        self.logger = logger
        self.path_to_store_logs = config['path_to_store_logs']
        pathlib.Path(self.path_to_store_logs).mkdir(parents=True, exist_ok=True)
        print('>> Successful')
        print('> Initialization is completed')

    def run_episodes(self, n_episodes: int):

        print('> Run is starting ...')
        self.reward_per_episode = 0
        self.steps_per_episode = 0
        self.total_steps = 0
        episode = 0

        pbar = tqdm.tqdm(total=n_episodes)
        while episode <= n_episodes:
            reward, obs, is_first = self.environment.observe()
            if is_first:
                if episode != 0:
                    pbar.update(1)
                episode += 1
                self.total_steps += self.steps_per_episode - 1

                self.steps_per_episode = 0
                self.reward_per_episode = 0

                self.agent.reset()

            self.reward_per_episode += reward
            current_action = self.agent.act(reward, obs, is_first)
            self.environment.act(current_action)
            self.steps_per_episode += 1
        pbar.close()

        plt.imshow(self.environment.env.render_rgb()[0])
        plt.show()
        print('> Run is finished.')
