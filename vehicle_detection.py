from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 nano model (fastest, downloads automatically if not present)
model = YOLO("yolov8n.pt")

# Open traffic video (replace with 0 for webcam)
cap = cv2.VideoCapture("traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Annotated frame with boxes + labels
    annotated_frame = results[0].plot()

    # Vehicle classes we care about
    vehicle_classes = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
    vehicle_count = {cls: 0 for cls in vehicle_classes}

    # Loop detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in vehicle_count:
            vehicle_count[cls_name] += 1

    # Overlay counts on screen
    y_offset = 30
    for cls, count in vehicle_count.items():
        cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        y_offset += 30

    # Show
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Quit with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
