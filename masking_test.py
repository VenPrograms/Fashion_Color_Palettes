from ultralytics import YOLO
import cv2
import numpy as np

# Load the model you just downloaded
model = YOLO('yolo26n-seg.pt')

# 1. Load your real image
img_path = "testimg.jpg"  # <-- Make sure you have an image with this name!
img = cv2.imread(img_path)

# 2. Predict
results = model.predict(img, conf=0.25) # conf=0.25 ignores 'uncertain' detections

# 3. View the results
# This will open a window showing the image with colorful masks over the clothes
for r in results:

    names = r.names 
    
    for i, box in enumerate(r.boxes):
        class_id = int(box.cls[0])
        label = names[class_id]
        
        # If the model found a specific clothing item (like 'tie' or 'handbag')
        # we extract its mask. If it only found 'person', we have to be 
        # more surgical.
        if label == "person":
            print("Found a person, but we need specific clothing labels!")
            # This is where DeepFashion2 will save us later!

            
    im_array = r.plot()  # plots the masks/boxes on the image
    cv2.imshow("YOLO26 Detection", im_array)
    cv2.waitKey(0) # Press any key to close the window

cv2.destroyAllWindows()