from typing import Dict, List
import math
import numpy as np

def is_image_space_channels_first(obs):
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(obs.shape).item()
    if smallest_dimension == 1:
        print("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0

# Base class of agent indicators
# @agent_indicator_mapping: This maps each agent in the environment to its corresponding indicator.
#   In [apply] function, we can apply the proper transformation according to this mapping.
class AgentIndicator:
    def __init__(self, env, agent_indicator_mapping: Dict[str, str]):
        self.env = env
        for agent in agent_indicator_mapping.keys():
            assert agent in env.possible_agents
        self.agent_indicator_mapping = agent_indicator_mapping

    def apply(self, obs, obs_space, agent):
        pass

# Invert an agent's observation by subtracting it from the maximum observable value.
class InvertColorIndicator(AgentIndicator):
    def __init__(self, env, invert_agents: List[str]):
        agent_indicator_mapping = {}
        for agent in env.possible_agents:
            agent_indicator_mapping[agent] = 'invert' if agent in invert_agents else 'none'
        super().__init__(env, agent_indicator_mapping)

    def apply(self, obs, obs_space, agent):
        high = obs_space.high.copy()
        if len(high.shape) == 2:
            high = high[:,:,None]
        return high - obs if self.agent_indicator_mapping[agent] is 'invert' else obs

# Binary representation of an agent by its index in the env.
class BinaryIndexIndicator(AgentIndicator):
    def __init__(self, env):
        super().__init__(env, {})

    def apply(self, obs, obs_space, agent):
        num_agents = len(self.env.possible_agents)
        num_channels = math.ceil(math.log2(num_agents))
        index = self.env.possible_agents.index(agent)
        if is_image_space_channels_first(obs):
            indicator = np.zeros((num_channels,) + obs.shape[1:])
            for i in range(num_channels):
                indicator[i,:,:] = obs_space.high * (index // math.pow(2, num_channels - i - 1))
                index = index % math.pow(2, num_channels - i - 1)
        else:
            indicator = np.zeros(obs.shape[:2] + (num_channels,))
            for i in range(num_channels):
                indicator[:,:,i] = obs_space.high * (index // math.pow(2, num_channels - i - 1))
                index = index % math.pow(2, num_channels - i - 1)
        return indicator

# Use different geometric pattern to represent different type of agent
class GeometricPatternIndicator(AgentIndicator):
    # @agent_groups: List of groups of agents that should have same geometric pattern to represent
    #   e.g. [[agentA, agentB], [agentC], [agentD]] :   Agent A and B have same pattern. 
    #                                                   Agent C and D each has different patterns to represent them.    
    def __init__(self, env, agent_groups: List[List[str]]):
        self.num_patterns = 0
        agent_indicator_mapping = {}
        for group in agent_groups:
            for agent in group:
                assert agent in env.possible_agents
                agent_indicator_mapping[agent] = str(self.num_patterns)
            self.num_patterns += 1
        for agent in env.possible_agents:
            if agent not in agent_indicator_mapping.keys():
                agent_indicator_mapping[agent] = str(self.num_patterns)
                self.num_patterns += 1
        super().__init__(env, agent_indicator_mapping)
        self.build_patterns(env)

    def build_patterns(self, env):
        interval = 1.0 / (self.num_patterns - 1)
        self.patterns = {}
        for agent in env.possible_agents:
            pattern = env.observation_spaces[agent].high.copy()
            obs_shape = pattern.shape

            t = int(self.agent_indicator_mapping[agent])
            for i in range(obs_shape[0]):
                for j in range(obs_shape[1]):
                    c = t * interval
                    c = 1.0 if c > 1.0 else c
                    pattern[i][j] *= c
                    t += 1
                    t = 0 if t == self.num_patterns else t
            self.patterns[agent] = pattern

    def apply(self, obs, obs_space, agent):
        return self.patterns[agent]

class AgentIndicatorWrapper:
    def __init__(self, use_original_obs=True):
        self.use_original_obs = use_original_obs
        self.indicators: List[AgentIndicator] = []

    def add_indicator(self, indicator: AgentIndicator):
        self.indicators.append(indicator)

    def apply(self, obs, obs_space, agent):
        nobs = obs.copy()
        if len(nobs.shape) == 2:
            nobs = nobs[:,:,None]
        res = nobs.copy() if self.use_original_obs else None
        for indicator in self.indicators:
            ind = indicator.apply(nobs, obs_space, agent)
            if len(ind.shape) == 2:
                ind = ind[:,:,None]
            res = ind if res is None else np.concatenate([res, ind], axis=0 if is_image_space_channels_first(nobs) else 2)
        return res