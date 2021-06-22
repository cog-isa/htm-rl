from htm_rl.envs.biogwlab.environment import Environment
from htm_rl.envs.env import Wrapper


class AgentPositionProvider(Wrapper):
    root_env: Environment

    def get_info(self) -> dict:
        info = super(AgentPositionProvider, self).get_info()
        info['agent_position'] = self.root_env.agent.position
        return info


