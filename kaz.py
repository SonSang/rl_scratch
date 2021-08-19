from stable_baselines3 import PPO
from pettingzoo.butterfly import knights_archers_zombies_v7
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

n_evaluations = 20
n_agents = 4
n_envs = 4
n_timesteps = 1e7

def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env

env = knights_archers_zombies_v7.parallel_env()
env = ss.black_death_v2(env)
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
#env = ss.observation_lambda_v0(env, invert_agent_indication)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)
env = image_transpose(env)

eval_env = knights_archers_zombies_v7.parallel_env()
eval_env = ss.black_death_v2(eval_env)
eval_env = ss.color_reduction_v0(eval_env, mode='B')
eval_env = ss.resize_v0(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 3)
#eval_env = ss.observation_lambda_v0(eval_env, invert_agent_indication)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class='stable_baselines3')
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs*n_agents), 1)

model = PPO("CnnPolicy", env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/kaz_v7/', log_path='./logs/kaz_v7/', eval_freq=eval_freq, deterministic=True, render=False)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)

model = PPO.load("./logs/kaz_v7/best_model")
evaluations = np.load('./logs/kaz_v7/evaluations.npz')
timesteps = evaluations['timesteps']
rewards = np.array(evaluations['results'])
rewards = rewards.mean(axis = 1)

# draw learning curve
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Rewards")
plt.savefig("./logs/kaz_v7/learning_curve.png")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(mean_reward)
print(std_reward)

render_env = knights_archers_zombies_v7.env()
render_env = ss.color_reduction_v0(render_env, mode='B')
render_env = ss.resize_v0(render_env, x_size=84, y_size=84)
render_env = ss.frame_stack_v1(render_env, 3)
#render_env = ss.observation_lambda_v0(render_env, invert_agent_indication)

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
            obs_list.append(render_env.render(mode='rgb_array'))
    render_env.close()
    break

print('Writing gif')
imgs = [Image.fromarray(img) for img in obs_list]
imgs[0].save('./logs/kaz_v7/kaz_v7.gif', save_all=True, append_images=imgs[1:], duration=60, loop=0)