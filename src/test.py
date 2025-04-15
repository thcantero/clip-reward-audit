from stable_baselines3 import SAC
from robot_env import RobotEnv

env = RobotEnv()
model = SAC.load("sac_reactor_x150")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()