from htm_rl.common.sdr import SparseSdr


class Agent:

    @property
    def name(self):
        raise NotImplementedError

    def act(self, reward: float, state: SparseSdr, first: bool):
        raise NotImplementedError

    def get_info(self) -> dict:
        return dict()


class Wrapper(Agent):
    agent: Agent
    root_agent: Agent

    def __init__(self, agent: Agent):
        self.agent = agent
        self.root_agent = unwrap(agent)

    @property
    def name(self):
        return self.agent.name

    def act(self, reward: float, state: SparseSdr, first: bool):
        self.agent.act(reward, state, first)

    def get_info(self) -> dict:
        return self.agent.get_info()


def unwrap(agent):
    while isinstance(agent, Wrapper):
        agent = agent.agent
    return agent
