from utils import *

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2

if __name__ == "__main__":
    env = make_vec_env_mario('SuperMarioBros-Nes', n_envs=4)  # Makes 4 environments to use with PPO
    env = VecFrameStack(env, n_stack=4)  # Inputs 4 frames instead of 1 to give context to NN

    # model = PPO2.load("ppo2_mario")

    # Make sure to edit the policies.py in stable-baselines as said
    model = PPO2(CnnPolicy, env, verbose=1, learning_rate=0.0005, n_steps=512, nminibatches=16, noptepochs=10)
    model.learn(total_timesteps=1000000)

    # model.save("ppo2_mario")

    # Tests model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
