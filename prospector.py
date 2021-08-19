from stable_baselines3 import PPO
from pettingzoo.butterfly import prospector_v4
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

import numpy as np
from array2gif import write_gif

n_evaluations = 20
n_agents = 7
n_timesteps = 1e7

def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env

resize_size = 84
env = prospector_v4.parallel_env()
env = ss.resize_v0(env, x_size=resize_size, y_size=resize_size, linear_interp=True)
env = ss.color_reduction_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pad_observations_v0(env)
env = ss.frame_stack_v1(env, 3)
#env = ss.dtype_v0(env, np.float32)
#env = ss.normalize_obs_v0(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = VecMonitor(env)
env = image_transpose(env)

eval_env = prospector_v4.parallel_env()
eval_env = ss.resize_v0(eval_env, x_size=resize_size, y_size=resize_size, linear_interp=True)
eval_env = ss.color_reduction_v0(eval_env)
eval_env = ss.pad_action_space_v0(eval_env)
eval_env = ss.pad_observations_v0(eval_env)
eval_env = ss.frame_stack_v1(eval_env, 3)
#eval_env = ss.dtype_v0(eval_env, np.float32)
#eval_env = ss.normalize_obs_v0(eval_env)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_agents), 1)

model = PPO("CnnPolicy", env, verbose=3, gamma=0.99, n_steps=256, ent_coef=0.01, learning_rate=0.0001, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=5000)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/prospector_v4/', log_path='./logs/prospector_v4/', eval_freq=eval_freq, deterministic=True, render=False)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)

model = PPO.load("./logs/prospector_v4/best_model")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print("Mean reward: {}".format(mean_reward))
print("Std reward: {}".format(std_reward))

render_env = prospector_v4.env()
render_env = ss.resize_v0(render_env, x_size=resize_size, y_size=resize_size, linear_interp=True)
render_env = ss.color_reduction_v0(render_env)
render_env = ss.pad_action_space_v0(render_env)
render_env = ss.pad_observations_v0(render_env)
render_env = ss.frame_stack_v1(render_env, 3)

obs_list = []
i = 0
render_env.reset()

while True:
    for agent in render_env.agent_iter():
        observation, _, done, _ = render_env.last()
        action = model.predict(observation, deterministic=True)[0] if not done else None

        render_env.step(action)
        i += 1
        if i % (len(render_env.possible_agents)) == 0:
            obs_list.append(np.transpose(render_env.render(mode='rgb_array'), axes=(1, 0, 2)))
    render_env.close()
    break

print('Writing gif')
write_gif(obs_list, './logs/prospector_v4/prospector_v4.gif', fps=15)
