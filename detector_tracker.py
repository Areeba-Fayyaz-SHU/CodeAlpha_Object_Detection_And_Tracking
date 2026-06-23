import cv2
from ultralytics import YOLO

def run_object_tracking(source_path=0):
    """
    Runs real-time object detection and tracking using YOLO.
    source_path: 0 for webcam, or 'path/to/video.mp4' for a video file.
    """
    # 1. Load a pre-trained YOLO model (nano version is great for real-time inference)
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")  # or "yolo11n.pt"

    # 2. Set up video capture (Webcam or Video File)
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
    print(f"Error: Could not open video source: {source_path}")
    return

    print("Starting stream. Press 'q' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or failed to read frame.")
            break

        # 3 & 4. Process frame with YOLO built-in ByteTRACK/BoT-SORT tracker
        # persist=True ensures the model maintains tracking IDs across frames
        results = model.track(frame, persist=True, verbose=False)

        # 5. Extract and display bounding boxes, labels, and tracking IDs
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding box coordinates
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class indices
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Unique tracking IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                label = model.names[class_id]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Format label with tracking ID and confidence
                text = f"ID: {track_id} {label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output real-time frame
        cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # To run on your traffic video:
    run_object_tracking(source_path="traffic_video.mp4")