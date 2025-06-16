import base64
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth

def test_vertex_ai_endpoint(image_path, project_id, endpoint_id, location="europe-west2"):
    credentials, project = google.auth.default()
    auth_req = Request()
    credentials.refresh(auth_req)
    
    endpoint_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "instances": [{"b64": image_b64}]
    }
    
    print(f"Testing endpoint: {endpoint_url}")
    response = requests.post(endpoint_url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print(json.dumps(result, indent=2))
        
        if "predictions" in result and len(result["predictions"]) > 0:
            prediction = result["predictions"][0]
            total_lanes = prediction.get("total_lanes", 0)
            print(f"\n📊 Summary: Found {total_lanes} lanes")
            
            for lane in prediction.get("lanes", []):
                print(f"  - {lane['type']}: confidence {lane['confidence']:.2f}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    PROJECT_ID = "ai-lane-detect"
    ENDPOINT_ID = "4431942255565078528"
    IMAGE_PATH = "/Users/naomi/Downloads/projects/GelLaneDetectionMinimal/3000..jpg"
    
    test_vertex_ai_endpoint(IMAGE_PATH, PROJECT_ID, ENDPOINT_ID)
