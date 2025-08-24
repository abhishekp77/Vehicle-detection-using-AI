from ultralytics import YOLO
import cv2
import csv
import time
import os


SOURCE = "traffic2.mp4"  # Change to 0 for webcam
OUTPUT_VIDEO = "output_traffic.avi"


model = YOLO("yolov8s.pt")

# Open video source
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise Exception(f"❌ Cannot open video source: {SOURCE}")

# Video writer for saving output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Vehicle classes (COCO dataset IDs)
vehicle_classes = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# Accumulated counts
vehicle_count = {name: 0 for name in vehicle_classes.values()}

# Create CSV log file
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_path = f"vehicle_counts_{timestamp}.csv"
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "vehicle_type", "cumulative_count"])

# Counting line (horizontal line at 2/3 height)
line_y = int(height * 0.66)

# Store IDs of already counted vehicles
counted_ids = set()

# ---------------------------
# Processing loop
# ---------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection + tracking only for vehicles
    results = model.track(
        frame,
        persist=True,
        imgsz=640,
        conf=0.4,
        device="cuda",  # change to "cpu" if no GPU
        classes=list(vehicle_classes.keys())
    )

    annotated = results[0].plot()

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            track_id = int(box.id[0])  # unique ID for each vehicle
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if vehicle crossed the counting line
            if cy > line_y - 5 and cy < line_y + 5:  
                if track_id not in counted_ids:  
                    cls_name = vehicle_classes[cls_id]
                    vehicle_count[cls_name] += 1
                    counted_ids.add(track_id)

                    # Save in CSV
                    csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), cls_name, vehicle_count[cls_name]])

            # Draw center point
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    # Draw counting line
    cv2.line(annotated, (0, line_y), (width, line_y), (255, 0, 0), 2)

    # Overlay accumulated counts
    y_offset = 30
    for cls, count in vehicle_count.items():
        cv2.putText(
            annotated,
            f"{cls}: {count}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_offset += 30

    # Show window
    cv2.imshow("Vehicle Counter (Line Crossing)", annotated)

    # Save to output video
    out.write(annotated)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------
# Cleanup
# ---------------------------
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"\n✅ Finished processing.")
print(f"📂 Output video saved as: {os.path.abspath(OUTPUT_VIDEO)}")
print(f"📂 CSV log saved as: {os.path.abspath(csv_path)}")
