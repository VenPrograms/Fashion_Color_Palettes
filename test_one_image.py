from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

# 1. Initialize the Client
# This will handle the 'weights' for you automatically
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zboym1fZcR8kXWX1nlZ1"
)

# 2. Run Inference
# Use the model ID from the Roboflow Universe page (e.g., 'deepfashion2-m-10k/1')
model_id = "deepfashion2-m-10k/2"
result = CLIENT.infer("testimg.jpg", model_id=model_id)

# 3. Extract the Masks
# Roboflow's result format is slightly different than Ultralytics'
# It gives you a list of predictions with points for the polygon mask
predictions = result['predictions']

for pred in predictions:
    label = pred['class']
    points = pred['points'] # These are the polygon coordinates
    
    # Logic: If it's a 'top' or 'dress', we extract it...
    print(f"AI found: {label}")