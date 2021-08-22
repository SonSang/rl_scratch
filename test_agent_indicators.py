from numpy import uint8
from pettingzoo.butterfly import prospector_v4
import supersuit as ss

from PIL import Image
import agent_indicators

from matplotlib import pyplot as plt
import numpy as np

log_dir = './agent_indicator_images'

resize_size = 84
render_env = prospector_v4.env()
render_env = ss.resize_v0(render_env, x_size=resize_size, y_size=resize_size, linear_interp=True)
render_env = ss.color_reduction_v0(render_env)
render_env = ss.pad_action_space_v0(render_env)
render_env = ss.pad_observations_v0(render_env)
#render_env = ss.frame_stack_v1(render_env, 3)

agent_indicator_wrapper = agent_indicators.AgentIndicatorWrapper()
agent_indicator_wrapper.add_indicator(agent_indicators.InvertColorIndicator(render_env, ['prospector_0', 'prospector_1', 'prospector_2', 'prospector_3']))
agent_indicator_wrapper.add_indicator(agent_indicators.BinaryIndexIndicator(render_env))
agent_indicator_wrapper.add_indicator(agent_indicators.GeometricPatternIndicator(render_env, [['prospector_0', 'prospector_1', 'prospector_2', 'prospector_3'], ['banker_0', 'banker_1', 'banker_2']]))
agent_indicator_wrapper.add_indicator(agent_indicators.GeometricPatternIndicator(render_env, [[agent] for agent in render_env.possible_agents]))
render_env = ss.observation_lambda_v0(render_env, agent_indicator_wrapper.apply)

render_env.reset()
#Image.fromarray(render_env.render(mode='rgb_array')).save(log_dir + "/scene.png")

cols = ['original', 'inverted', 'binary\n(index)', 'binary\n(index)', 'binary\n(index)', 'geometric\n(type)', 'geometric\n(index)']
rows = [agent for agent in render_env.possible_agents]
fig, axs = plt.subplots(7, 7)
for ax, col in zip(axs[0], cols):
    ax.set_title(col, size='x-small')
for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=0, size='x-small')
ax.set_yticklabels([])
ax.set_xticklabels([])

cnt = 0
for agent in render_env.agent_iter():
    observation, _, done, _ = render_env.last()

    # render observation itself
    # render channel by channel
    '''
    if len(observation.shape) == 3:
        for i in range(observation.shape[2]):
            image = observation[:,:,i]
            image = image.astype(uint8)
            Image.fromarray(image).save(log_dir + "/{}_channel_{}.png".format(agent, i))
    else:
        Image.fromarray(observation).save(log_dir + "/{}.png".format(agent))
    break
    '''
    for i in range(observation.shape[2]):
        image = observation[:,:,i]
        image = image.astype(uint8)
        axs[cnt,i].imshow(image, cmap='gray', vmin=0, vmax=255)
        axs[cnt,i].get_xaxis().set_visible = False
        axs[cnt,i].get_yaxis().set_visible = False
    cnt += 1
    if cnt == 7:
        break
    render_env.step(np.array([0, 0, 0]))
plt.savefig(log_dir + '/indicators.png')
        
render_env.close()