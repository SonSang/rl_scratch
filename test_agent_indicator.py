from numpy import uint8
from pettingzoo.butterfly import prospector_v4
import supersuit as ss
from matplotlib import pyplot as plt
import numpy as np
import argparse

import agent_indicator

log_dir = './agent_indicator_images'

parser = argparse.ArgumentParser()
parser.add_argument("--indicator-name", type=str, default="invert", help = "Name of agent indicator to use", choices=['invert', 'binary', 'geometric'])
parser.add_argument("--replace-obs", action="store_true", help = "Whether or not to replace obs with agent indicator")
args = parser.parse_args()

resize_size = 84
render_env = prospector_v4.env()
render_env = ss.resize_v0(render_env, x_size=resize_size, y_size=resize_size, linear_interp=True)
render_env = ss.color_reduction_v0(render_env)
render_env = ss.pad_action_space_v0(render_env)
render_env = ss.pad_observations_v0(render_env)
#render_env = ss.frame_stack_v1(render_env, 3)

indicator_type = 'prospector'
if args.indicator_name == 'invert':
    indicator = agent_indicator.InvertColorIndicator(render_env, indicator_type)
elif args.indicator_name == 'binary':
    indicator = agent_indicator.BinaryIndicator(render_env, indicator_type)
elif args.indicator_name == 'geometric':
    indicator = agent_indicator.GeometricPatternIndicator(render_env, indicator_type)
agent_indicator_wrapper = agent_indicator.AgentIndicatorWrapper(indicator, not args.replace_obs)
render_env = ss.observation_lambda_v0(render_env, agent_indicator_wrapper.apply)
render_env.reset()

rows = []
if not args.replace_obs:
    rows.append('original')
rows.append('indicator\n({})'.format(args.indicator_name))
cols = [agent for agent in render_env.possible_agents]

fig, axs = plt.subplots(len(rows), len(cols))
if len(axs.shape) == 1:
    for ax, col in zip(axs, cols):
        ax.set_title(col, size='x-small')
else:
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, size='x-small')
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='x-small')

done_agents = []
cnt = 0
for agent in render_env.agent_iter():
    observation, _, done, _ = render_env.last()

    if agent in done_agents:
        break
    # render channel by channel
    for i in range(observation.shape[2]):
        image = observation[:,:,i]
        image = image.astype(uint8)
        if len(axs.shape) == 1:
            axs[cnt].imshow(image, cmap='gray', vmin=0, vmax=255)
        else:
            axs[i, cnt].imshow(image, cmap='gray', vmin=0, vmax=255)
    cnt += 1
    done_agents.append(agent)
    render_env.step(np.array([0, 0, 0]))
plt.savefig(log_dir + '/indicators_{}_{}.png'.format(args.indicator_name, 'replace' if args.replace_obs else 'noreplace'))
        
render_env.close()