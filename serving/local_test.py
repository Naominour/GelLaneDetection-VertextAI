import requests
import base64
import json

# Test health
response = requests.get("http://localhost:8080/health")
print("Health:", response.json())

# Test prediction
with open("/Users/naomi/Downloads/projects/GelLaneDetectionMinimal/3000..jpg", 'rb') as f:
    image_bytes = f.read()

image_b64 = base64.b64encode(image_bytes).decode('utf-8')

data = {
    "instances": [{"b64": image_b64}]
}

response = requests.post("http://localhost:8080/predict", json=data)
print("Prediction:", response.json())
