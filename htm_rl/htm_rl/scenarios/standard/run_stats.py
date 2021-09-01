import numpy as np


class RunStats:
    steps: list[int]
    rewards: list[float]
    times: list[float]

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.times = []

    def append_stats(self, steps, total_reward, elapsed_time):
        self.steps.append(steps)
        self.rewards.append(total_reward)
        self.times.append(elapsed_time)

    def print_results(self):
        steps = np.array(self.steps)
        avg_len = steps.mean()
        last_10_pct = steps.shape[0] // 10
        last10_avg_len = steps[-last_10_pct:].mean()
        avg_reward = np.array(self.rewards).mean()
        avg_time = np.array(self.times).mean()
        elapsed = np.array(self.times).sum()
        print(
            f'Len10: {last10_avg_len: .2f}  Len: {avg_len: .2f}  '
            f'R: {avg_reward: .5f}  '
            f'EpT: {avg_time: .6f}  TotT: {elapsed: .6f}'
        )

    @staticmethod
    def aggregate_stats(agent_results):
        results = RunStats()
        results.steps = np.mean([res.steps for res in agent_results], axis=0)
        results.times = np.mean([res.times for res in agent_results], axis=0)
        results.rewards = np.mean([res.rewards for res in agent_results], axis=0)
        return results