import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import cv2

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        # Connect to PyBullet
        p.connect(p.DIRECT)
        #p.connect(p.GUI)  # Use p.DIRECT for headless
        p.setGravity(0, 0, -9.8)
        
        # Define action/observation spaces (customize robot)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # Joint torques
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)  # RGB image

    def step(self, action):
        # Apply action to robot joints
        # Capture image observation
        obs = self._get_observation()

        cv2.imwrite("logs/output/success.png", obs)  # Save when task is successful
        cv2.imwrite("logs/output/failed.png", obs)   # Save when task fails

        # Calculate reward (we’ll add CLIP later)
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def _get_observation(self):
        # Render robot camera view (adjust camera position)
        view_matrix = p.computeViewMatrix([0.5, 0, 1], [0, 0, 0], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
        _, _, rgb, _, _ = p.getCameraImage(224, 224, view_matrix, proj_matrix)
        return rgb[:, :, :3]  # Shape (224, 224, 3)

    def reset(self):
        p.resetSimulation()
        
        # Load robot URDF after connecting to PyBullet
        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "reactor_x150",
            "urdf",
            "reactor_x150.urdf"
        )
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        
        return self._get_observation()