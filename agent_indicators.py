from typing import Dict
import math
import numpy as np

# Base class of agent indicators
# @agent_indicator_mapping: This maps each agent in the environment to its corresponding indicator.
#   In [apply] function, we can apply the proper transformation according to this mapping.
class AgentIndicator:
    def __init__(self, env, agent_indicator_mapping: Dict[str, str]):
        self.env = env
        assert agent_indicator_mapping.keys() in env.possible_agents
        self.agent_indicator_mapping = agent_indicator_mapping

    def apply(self, obs, obs_space, agent):
        pass

# Invert an agent's observation by subtracting it from the maximum observable value.
class InvertColorIndicator(AgentIndicator):
    def __init__(self, env, agent_indicator_mapping: Dict[str, str]):
        super.__init__(InvertColorIndicator, env, agent_indicator_mapping)

    def apply(self, obs, obs_space, agent):
        return obs_space.high - obs if self.agent_indicator_mapping[agent] is 'invert' else obs

# Binary representation of an agent by its index in the env.
class BinaryIndexIndicator(AgentIndicator):
    def __init__(self, env):
        super.__init__(BinaryIndexIndicator, env, None)

    def apply(self, obs, obs_space, agent):
        num_agents = len(self.env.possible_agents)
        num_channels = math.ceil(math.log2(num_agents))
        index = self.env.possible_agents.index(agent)
        # Assume [W, H, C] order
        indicator = np.zeros(obs.shape[:2] + (num_channels,))
        for i in range(num_channels):
            indicator[:,:,i] = obs_space.high * (index // math.pow(2, num_channels - i - 1))
            index = index % math.pow(2, num_channels - i - 1)
        return indicator