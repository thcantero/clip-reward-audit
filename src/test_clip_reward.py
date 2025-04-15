# test_clip_reward.py
import cv2
from robot_env import RobotEnv
from clip_reward import CLIPReward

env = RobotEnv()
obs = env.reset()

# Step once so success.png is created
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

# Now the environment wrote success.png
img = cv2.imread("logs/output/success.png")
if img is None:
    raise FileNotFoundError("success.png not found. Check your path or working directory!")

reward_model = CLIPReward(goal_text="some text prompt")
success_reward = reward_model.compute_reward(img)
print("Reward:", success_reward)
