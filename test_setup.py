import torch
from ultralytics import YOLO
import numpy as np

print("--- Starting Environment Test ---")

# 1. Load the YOLO26 Nano Segmentation model
# This will auto-download the file to your current folder if it's missing
try:
    model = YOLO('yolo26n-seg.pt')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model failed to load: {e}")
    exit()

# 2. Create a 'Dummy' Image
# A 640x640 image with 3 color channels (RGB) filled with zeros (black)
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

# 3. Run a Test Prediction
# We use 'conf=0.1' just to ensure the internal math triggers
try:
    results = model.predict(dummy_img, imgsz=640, verbose=False)
    print("Inference complete. The engine is running!")
    
    # Check if GPU is being used
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device}")
    
except Exception as e:
    print(f"Inference failed: {e}")

print("--- Test Finished ---")