from utils import *

from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.policies import CnnPolicy
from ppo3 import PPO3

LOAD_MODEL = True

if __name__ == "__main__":
    env = make_vec_env_mario('SuperMarioBros-Nes', n_envs=4, seed=123)  # Makes 4 environments to use with PPO
    env = VecFrameStack(env, n_stack=4)  # Inputs 4 frames instead of 1 to give context to NN
    env = VecNormalize(env, norm_obs=False, norm_reward=True,  # Normalize reward from environments
                       clip_obs=10.)

    if LOAD_MODEL:
        model = PPO3.load("ppo2_mario", tensorboard_log="./a2c_cartpole_tensorboard/")
    else:
        model = PPO3(CnnPolicy, env, verbose=1, learning_rate=0.00025, n_steps=2048, nminibatches=1, noptepochs=4,
                     cliprange=0.1, tensorboard_log="./a2c_cartpole_tensorboard/")

    # Make sure to edit the policies.py in stable-baselines as said in README.
    model.set_env(env)
    model.learn(total_timesteps=9000000)
    model.save("ppo2_mario")

    # Tests model
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
