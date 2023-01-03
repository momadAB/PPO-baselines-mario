from utils import *

from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines import PPO2
from ppo3 import PPO3

if __name__ == "__main__":
    env = make_vec_env_mario('SuperMarioBros-Nes', n_envs=4, seed=123)  # Makes 4 environments to use with PPO
    env = VecFrameStack(env, n_stack=4)  # Inputs 4 frames instead of 1 to give context to NN

    # model = PPO3.load("ppo2_mario245760", tensorboard_log="./a2c_cartpole_tensorboard/")
    # model = PPO3.load("ppo2_mario(earlystop)", tensorboard_log="./a2c_cartpole_tensorboard/")
    # Stop training when the model reaches the reward threshold
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
    # eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    env = VecNormalize(env, norm_obs=False, norm_reward=True,
                       clip_obs=10.)
    # # Make sure to edit the policies.py in stable-baselines as said

    model = PPO3(CnnPolicy, env, verbose=1, learning_rate=0.0005, n_steps=2048, nminibatches=4, noptepochs=14,
                 cliprange=0.05, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.set_env(env)
    model.learn(total_timesteps=2048000)
    model.save("ppo2_mario")

    # Tests model
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # if env.buf_dones[0] == True or env.buf_dones[1] == True or env.buf_dones[2] == True or env.buf_dones[3] == True:
        #     print(env.buf_dones)
        env.render()

    env.close()