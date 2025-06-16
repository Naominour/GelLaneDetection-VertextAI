import cv2
import numpy as np

import albumentations as A
import torchvision.transforms.functional as TF



class InferencePreprocessor:
    
    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size
        self.transform = self.get_inference_transforms()
    
    def get_inference_transforms(self):
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1], interpolation=cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ])
    
    def color_normalise(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)   
        lab = cv2.merge((l, a, b))

        normalised = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return normalised
    
    def preprocess_image(self, image):
        normalized_image = self.color_normalise(image)
        transformed = self.transform(image=normalized_image)
        return transformed['image']
