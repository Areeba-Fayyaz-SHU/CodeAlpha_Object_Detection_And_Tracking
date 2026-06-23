# CodeAlpha - Object Detection and Tracking

A real-time Object Detection and Tracking system built as part of the CodeAlpha Artificial Intelligence Internship. This project processes video inputs frame-by-frame to accurately localize objects with bounding boxes, identify their classifications, and maintain consistent unique identity tracking across frames using state-of-the-art computer vision models.

## Features
- **Real-Time Detection:** Utilizes deep learning models to identify a wide range of objects instantly.
- **Persistent Object Tracking:** Integrates robust tracking algorithms (ByteTRACK/BoT-SORT via YOLO tracking API) to maintain unique tracking IDs for individual objects across frames.
- **Dynamic Input Processing:** Supports flexible video sourcing including direct real-time webcam streams or static video file analysis (`.mp4`).
- **Clean Visual Overlays:** Draws dynamic bounding boxes, tracking IDs, and class labels with confidence metrics on the live video feedback loop.

## Technologies Used
- Python
- OpenCV (`cv2`)
- Ultralytics YOLO Framework
- Deep Learning & Computer Vision Metrics

## How to Run Locally

1. Clone this repository or download the files.
2. Install the required dependencies:
```bash
   pip install ultralytics opencv-python
3. Run the tracking script:
      python detector_tracker.py
