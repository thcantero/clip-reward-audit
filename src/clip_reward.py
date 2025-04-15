import clip
import torch
from PIL import Image
import cv2
import numpy as np

class CLIPReward:
    def __init__(self, goal_text="a robot arm grasping a red block"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.goal_text = goal_text
        self.goal_features = self._encode_text(goal_text)

    def _encode_image(self, image):
        
        if isinstance(image, np.ndarray):
            # 1) Convert from BGR to RGB using OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2) Convert to PIL
            image = Image.fromarray(image_rgb)
        else:
        
            # If it's already a PIL Image, just ensure it's in RGB mode
            if not image.mode == "RGB":
                image = image.convert("RGB")

        # Now `image` is a PIL image in RGB mode
        preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(preprocessed)
        return features

    def _encode_text(self, text):
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(text_tokens)
        return features

    def compute_reward(self, image):
        image_features = self._encode_image(image)
        similarity = torch.cosine_similarity(image_features, self.goal_features, dim=1)
        return similarity.item()