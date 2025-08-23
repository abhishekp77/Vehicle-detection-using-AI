from ultralytics import YOLO
import cv2
import csv

# ---------------------------
# Setup
# ---------------------------
model = YOLO("yolov8n.pt")   # small, fast model
cap = cv2.VideoCapture(0)  # change to 0 for webcam

# Vehicle classes to track
vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
total_vehicle_count = {cls: 0 for cls in vehicle_classes}

# Tracking dictionary for previous positions
prev_positions = {}

# CSV file to save results
csv_file = open("vehicle_counts.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["vehicle_type", "count"])  # header

# ---------------------------
# Processing loop
# ---------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with tracking (assigns ID to each vehicle)
    results = model.track(frame, persist=True, imgsz=480, device="cpu")

    annotated_frame = results[0].plot()

    # Iterate over detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        # Only track relevant vehicles
        if cls_name not in vehicle_classes:
            continue

        obj_id = int(box.id[0]) if box.id is not None else None
        if obj_id is None:
            continue

        # Current Y position (to detect movement direction)
        y_center = int(box.xywh[0][1])

        # Check if object was seen before
        if obj_id in prev_positions:
            if y_center > prev_positions[obj_id] + 5:  # moving downward (approaching)
                # Count this vehicle once when it crosses the mid-line
                frame_mid = frame.shape[0] // 2
                if prev_positions[obj_id] < frame_mid <= y_center:
                    total_vehicle_count[cls_name] += 1
                    print(f"Counted {cls_name}, Total = {total_vehicle_count[cls_name]}")

                    # Save to CSV
                    csv_writer.writerow([cls_name, total_vehicle_count[cls_name]])

        # Update position
        prev_positions[obj_id] = y_center

    # Overlay accumulated counts
    y_offset = 30
    for cls, count in total_vehicle_count.items():
        cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    # Show video
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
