from robot_env import RobotEnv
import time

env = RobotEnv()
obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Random actions
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
    time.sleep(0.1)  # Slow down for visualization
