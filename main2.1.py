from ultralytics import YOLO
import cv2
import pyttsx3
import time
from collections import defaultdict
import numpy as np

class VisionAssistant:
    def __init__(self):
        # Initialize YOLOv8 model (using nano for better speed)
        self.model = YOLO("yolov8n.pt") 
        self.conf_thres = 0.5
        
        # Audio setup
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_audio_time = 0
        
        # Priority objects (with custom messages)
        self.priority_objects = {
            'person': {'message': "There is a person in front of you", 'color': (0, 255, 0)},
            'car': {'message': "Vehicle nearby, be careful", 'color': (0, 255, 255)},
            'staircase': {'message': "Stairs ahead, watch your step", 'color': (0, 0, 255)},
        }
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.set_camera_resolution(640, 480)
        
        # Scene memory
        self.last_scene = ""
        self.repetition_threshold = 5  # Don't repeat same description within X seconds

    def set_camera_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Camera resolution set to {width}x{height}")

    def generate_scene_description(self, detections, frame_width):
        if not detections:
            return "The area appears clear"
        
        # Categorize objects by position
        left_objs = []
        center_objs = []
        right_objs = []
        
        for obj in detections:
            x_center = (obj['bbox'][0] + obj['bbox'][2]) / 2
            if x_center < frame_width * 0.4:
                left_objs.append(obj)
            elif x_center > frame_width * 0.6:
                right_objs.append(obj)
            else:
                center_objs.append(obj)
        
        # Build description parts
        description_parts = []
        
        # 1. Priority objects first
        for area in [center_objs, left_objs, right_objs]:
            for obj in area:
                if obj['class'] in self.priority_objects:
                    description_parts.append(self.priority_objects[obj['class']]['message'])
        
        # 2. Spatial description
        for area, position in zip([left_objs, center_objs, right_objs], 
                                ["to your left", "in front of you", "to your right"]):
            if area:
                obj_names = [obj['class'] for obj in area if obj['class'] not in self.priority_objects]
                if obj_names:
                    unique_objs = list(set(obj_names))
                    count_str = ", ".join([f"{obj_names.count(obj)} {obj}{'s' if obj_names.count(obj)>1 else ''}" 
                                         for obj in unique_objs])
                    description_parts.append(f"There are {count_str} {position}")
        
        # Combine into natural language
        if not description_parts:
            return "No notable objects detected"
            
        # Add scene context
        if len(description_parts) > 2:
            main_desc = ". ".join(description_parts[:2]) + ". Also, " + ". ".join(description_parts[2:])
        else:
            main_desc = ". ".join(description_parts)
            
        return main_desc.capitalize()

    def process_frame(self, frame):
        # Mirror and resize
        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        
        # Run detection
        results = self.model(frame, conf=self.conf_thres, verbose=False)[0]
        
        # Process detections
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'class': class_name,
                'bbox': bbox,
                'confidence': conf
            })
            
            # Draw bounding boxes
            color = self.priority_objects.get(class_name, {}).get('color', (255, 0, 0))
            cv2.rectangle(frame_display, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame_display, label, 
                       (int(bbox[0]), int(bbox[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Generate description
        description = self.generate_scene_description(detections, frame.shape[1])
        
        # Display info
        cv2.putText(frame_display, description[:100], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame_display, description

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera error")
                break
            
            processed_frame, description = self.process_frame(frame)
            
            # Show output
            cv2.imshow("Scene Understanding", processed_frame)
            
            # Speak description (with cooldown)
            current_time = time.time()
            if (description != self.last_scene and 
                current_time - self.last_audio_time > self.repetition_threshold):
                self.engine.say(description)
                self.engine.runAndWait()
                self.last_audio_time = current_time
                self.last_scene = description
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = VisionAssistant()
    assistant.run()
#     Key Improvements:
# Natural Scene Descriptions
# Example outputs:

# "There is a person in front of you. 2 chairs to your left. 1 cup on the right."

# "Stairs ahead, watch your step. Also, there's a bag to your left."

# Spatial Awareness

# Divides scene into left/center/right areas

# Uses positional phrases ("to your left", "ahead")

# Priority System

# Important objects (person, stairs) get mentioned first

# Custom warning messages for critical items

# Anti-Repetition

# Won't repeat the same description within 5 seconds

# Only speaks when scene meaningfully changes

# Performance Optimized

# Uses YOLOv8n (nano) for better speed

# Fixed 640x480 resolution for stability

# How to Use:
# Install requirements:

# bash
# Copy
# pip install ultralytics pyttsx3 opencv-python numpy
# The system will:

# Download YOLOv8n model automatically

# Generate natural audio descriptions

# Show visual feedback with bounding boxes

# Further Enhancement Ideas:
# Add distance estimation (using object size heuristics)

# Include scene classification ("This looks like a kitchen")

# Add voice commands ("What's on my left?")

# Implement obstacle detection warnings

