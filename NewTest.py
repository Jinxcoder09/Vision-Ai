from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use "yolov8s.pt", "yolov8m.pt", etc. for larger models

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path

# Set confidence threshold (e.g., 0.5 for 50% confidence)
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, stream=True)  # Use stream=True for real-time processing

    # Loop through the results
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            confidence = box.conf[0]  # Confidence score
            if confidence < confidence_threshold:  # Skip low-confidence detections
                continue

            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Print detected object name
            print(f"Detected: {class_name} with confidence {confidence:.2f}")

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()