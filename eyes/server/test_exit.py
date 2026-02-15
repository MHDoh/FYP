import cv2
import numpy as np
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

# Video path (adjust to your environment)
video_path = r"C:\Users\mohdh\OneDrive\Desktop\channel with tower\scs.mp4"

# Load YOLOv11n model
model = YOLO(r"C:\Users\mohdh\OneDrive\Desktop\Karel\yolo11n.pt")

# Run tracking with ByteTrack (make sure "bytetrack.yaml" is available in the working directory or provide full path)
results = model.track(source=video_path, tracker=r"C:\Users\mohdh\Downloads\bytetrack.yaml", conf=0.5)

# Define door line (e.g., a horizontal line at y=400)
door_line_y = 400

# Dictionary to store previous vertical center for each track
track_history = {}

# Process each frame result
for result in results:
    # Retrieve the frame image. Depending on the version, the attribute might differ.
    # Here, we try to use result.orig_img if available, otherwise use the first image in result.imgs.
    if hasattr(result, 'orig_img'):
        frame = result.orig_img.copy()
    else:
        frame = result.imgs[0].copy()

    persons = []  # List to hold person detections: (bbox, conf, track_id)
    objects = []  # List for non-person objects

    # Process detections in this frame
    for box in result.boxes:
        # Convert box coordinates to integers
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        track_id = int(box.id.item()) if hasattr(box, "id") else None

        # Assume class 0 is "person" (COCO). All others are treated as potential objects.
        if cls == 0:
            persons.append((xyxy, conf, track_id))
        else:
            objects.append((xyxy, conf, cls))

    # Process each person detection for door crossing anomaly
    for bbox, conf, track_id in persons:
        x1, y1, x2, y2 = bbox
        current_center = (y1 + y2) / 2
        prev_center = track_history.get(track_id, current_center)

        # Check if the person crosses the door line (from above to below)
        if prev_center < door_line_y <= current_center:
            # Check for overlap with any object using IoU threshold of 0.3
            carrying = any(compute_iou(bbox, obj_bbox) > 0.3 for (obj_bbox, _, _) in objects)
            if not carrying:
                cv2.putText(frame, f"Anomaly! ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update the track history
        track_history[track_id] = current_center

        # Draw the person's bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw door line for visualization
    cv2.line(frame, (0, door_line_y), (frame.shape[1], door_line_y), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("HOI Anomaly Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
