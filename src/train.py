import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from robot_env import RobotEnv
from clip_reward import CLIPReward

# Initialize environment and reward model
env = RobotEnv()
reward_model = CLIPReward(goal_text="a robot arm grasping a red block")

# Wrap the environment to use CLIP rewards
class ClipRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = reward_model.compute_reward(obs)
        return obs, reward, done, info

env = ClipRewardWrapper(env)

# Train SAC agent
model = SAC("CnnPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100_000)

# Save the policy
model.save("sac_reactor_x150")