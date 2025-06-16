import os
import torch
import cv2
import numpy as np
from src.models.mask_rcnn import create_mask_rcnn_model
from inference_preprocessing import InferencePreprocessor
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import uvicorn

class LaneDetectionPredictor:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ["Background", "Lane", "EmptyLane"]
        self.detection_threshold = 0.7

        self.model = create_mask_rcnn_model(num_classes=3, pretrained=False)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
        assert os.path.exists("model.pth"), "model.pth not found!"
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = InferencePreprocessor(target_size=(1024, 1024))

    def predict(self, instances):
        predictions = []

        for instance in instances:
            try:
                if isinstance(instance, dict):
                    if 'b64' in instance:
                        import base64
                        image_data = base64.b64decode(instance['b64'])
                    elif 'image' in instance:
                        image_data = instance['image']
                    else:
                        image_data = list(instance.values())[0]
                else:
                    image_data = instance
                
                result = self._process_single_image(image_data)
                predictions.append(result)
                
            except Exception as e:
                predictions.append({
                    "error": str(e),
                    "total_lanes": 0,
                    "lanes": []
                })       
        return {"predictions": predictions}
    
    def _process_single_image(self, image_data):
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        processed_image = self.preprocessor.preprocess_image(image)
        img_tensor = processed_image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
        
        return self._format_prediction(prediction)
    
    def _format_prediction(self, prediction):
        boxes = prediction.get('boxes', torch.tensor([])).cpu().numpy()
        scores = prediction.get('scores', torch.tensor([])).cpu().numpy()
        labels = prediction.get('labels', torch.tensor([])).cpu().numpy()
        
        if len(scores) > 0:
            keep = scores >= self.detection_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        
        lanes = []
        if len(labels) > 0:
            lane_indices = np.where((labels == 1) | (labels == 2))[0]
            
            for i, idx in enumerate(lane_indices):
                box = boxes[idx]
                label = labels[idx]
                score = scores[idx]
                
                x1, y1, x2, y2 = box.astype(int)
                
                lanes.append({
                    "id": i + 1,
                    "type": self.class_names[label],
                    "confidence": float(score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area": int((x2 - x1) * (y2 - y1)),
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
        
        return {
            "total_lanes": len(lanes),
            "lanes": lanes
        }

class PredictionRequest(BaseModel):
    instances: List[Any]

app = FastAPI(title="Lane Detection API", version="1.0.0")

predictor = LaneDetectionPredictor()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predictor.predict(request.instances)
        return result
    except Exception as e:
        return {"error": str(e), "predictions": []}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
