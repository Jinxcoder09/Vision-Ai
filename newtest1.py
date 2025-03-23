from ultralytics import YOLO
import cv2

# Load a larger YOLOv8 model for better accuracy
model = YOLO("yolov8m.pt")  # Use "yolov8l.pt" for even better accuracy

# Open the webcam
cap = cv2.VideoCapture(0)

# Set confidence threshold
confidence_threshold = 0.6  # Increase this value to reduce false positives

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame (horizontal flip)
    mirrored_frame = cv2.flip(frame, 1)

    # Increase input resolution for better detection
    resized_frame = cv2.resize(mirrored_frame, (1280, 720))

    # Perform object detection on the resized frame
    results = model(resized_frame, stream=True, conf=confidence_threshold)

    # Loop through the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]  # Confidence score
            if confidence < confidence_threshold:  # Skip low-confidence detections
                continue

            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name

            # Draw bounding box and label
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Print detected object name
            print(f"Detected: {class_name} with confidence {confidence:.2f}")

    # Display the resized and mirrored frame
    cv2.imshow("Mirrored YOLOv8 Object Detection", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()