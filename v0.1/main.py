from utils import *

from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines import PPO2

if __name__ == "__main__":
    env = make_vec_env_mario('SuperMarioBros-Nes', n_envs=4)  # Makes 4 environments to use with PPO
    env = VecFrameStack(env, n_stack=4)  # Inputs 4 frames instead of 1 to give context to NN

    # model = PPO2.load("ppo2_mario")
    # Stop training when the model reaches the reward threshold
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
    # eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    #
    # # Make sure to edit the policies.py in stable-baselines as said
    model = PPO2(CnnPolicy, env, verbose=1, learning_rate=0.0005, n_steps=512, nminibatches=2, noptepochs=11,
                 cliprange=0.1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=1000000)

    model.save("ppo2_mario")

    # Tests model
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if env.buf_dones[0] == True or env.buf_dones[1] == True or env.buf_dones[2] == True or env.buf_dones[3] == True:
            print(env.buf_dones)
        env.render()

    env.close()
