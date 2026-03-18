# StylePalette: Color Analyzer using Computer Vision

A computer vision tool that identifies clothing items in photos and extracts a dominant color palette representing a user's uploaded images. Built as a test of my technical skills with exploring AI models.

## Key Features

* **Instance Segmentation**: Uses the `DeepFashion2` dataset with a YOLO model via Roboflow to trace exact polygon outlines of clothes, ensuring skin and background colors do not corrupt the data.
* **Global Clustering**: Aggregates pixel data across multiple uploaded images to find the top $X$ most dominant colors in an entire collection.
* **Performance Optimized**: 
    * **Dynamic Scaling**: I automatically resize high-res uploads to 800px to reduce latency.
    * **Representative Sampling**: I use random sampling (1000px per item) to maintain accuracy while preventing memory (RAM) crashes or unnecessary use of memory to store every single pixel in an item

## Technical Stack

* **Language**: Python 3.10+
* **UI Framework**: Gradio (for web deployment)
* **AI Inference**: Roboflow Inference SDK (ResNet/YOLO-based segmentation)
* **Image Processing**: OpenCV (Masking, Polygon filling, BGR-RGB conversion)
* **Math/ML**: NumPy & K-Means Clustering 

## The Pipeline

1.  **Image Pre-processing**: Images are loaded and scaled using `cv2.INTER_AREA` interpolation to maintain color integrity while reducing data size.
2.  **AI Detection**: The Roboflow model identifies clothing "points" (JSON schema).
3.  **Binary Masking**: I create a mask is created using `cv2.fillPoly`. I extract pixels using Boolean Indexing (`mask > 0`), this is much faster by using C++ operations instead of slow Python loops.
4.  **Clustering**: Now I feed my aggregated Master Pixel List into the K-Means algorithm.
    * **Stop Criteria**: 10 iterations or 1.0 epsilon (this CAN BE CHANGED, current values work well)
    * **Attempts**: 10 random restarts to ensure the optimal color centers are found. CAN BE CHANGED
5.  **Output**: Hex codes are generated and a proportional bar is rendered using NumPy slicing.

