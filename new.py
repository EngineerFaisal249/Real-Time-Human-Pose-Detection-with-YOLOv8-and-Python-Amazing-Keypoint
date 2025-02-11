import cv2
import cvzone
from ultralytics import YOLO
import numpy as np

# Load video and YOLO model
cap = cv2.VideoCapture(r'renadom.mp4')
model = YOLO(r'yolo11x-pose.pt')

# Video writer setup (output file path, codec, FPS, resolution)
output_path = r'E:\Real-Time-Human-Pose-Detection-with-YOLOv8-and-Python-Amazing-Keypoint-Visualizations--main\demo.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Use input video FPS or default to 30
frame_width = 640
frame_height = 720
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

# Processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End video processing

    # Resize frame for processing
    frame = cv2.resize(frame, (frame_width, frame_height))
    height, width = frame.shape[:2]
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Perform pose estimation
    results = model(frame)
    frame = results[0].plot()

    # Process keypoints
    for keypoints in results[0].keypoints.data:
        keypoints = keypoints.cpu().numpy()
        for i, keypoint in enumerate(keypoints):
            x, y, confidence = keypoint
            if confidence > 0.7:
                cv2.circle(blank_image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=1)
                cv2.putText(blank_image, f'{i}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Connect keypoints
        connections = [
            (3, 1), (1, 0), (0, 2), (2, 4), (1, 2), (4, 6), (3, 5),
            (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10),
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (5, 11), (6, 12)
        ]
        for part_a, part_b in connections:
            x1, y1, conf1 = keypoints[part_a]
            x2, y2, conf2 = keypoints[part_b]
            if conf1 > 0.5 and conf2 > 0.5:
                cv2.line(blank_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), thickness=2)

    # Stack and save frames
    output_frame = cvzone.stackImages([frame, blank_image], cols=2, scale=0.80)
    out.write(output_frame)  # Save the stacked frame to output video

    # Display output frame (optional, for testing)
    cv2.imshow('Output', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
