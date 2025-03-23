import cv2
import numpy as np
import torch
import json
import time
from ultralytics import YOLO

class ObjectDetectionModel:
    def __init__(self, config_path=None):
        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt")  # Using YOLOv8 nano version for efficiency

    def predict(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes
        return detections

class OnlineLearner:
    def __init__(self, config_path=None):
        self.running = False
    
    def start(self, model):
        self.running = True
        print("Online learner started (not implemented yet)")
    
    def stop(self):
        self.running = False
        print("Online learner stopped")

class DetectionUtils:
    @staticmethod
    def load_config(config_path):
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Config file '{config_path}' not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in '{config_path}'.")
            return {}
    
    @staticmethod
    def preprocess_frame(frame):
        return cv2.resize(frame, (640, 640))  # Resize for YOLO
    
    @staticmethod
    def non_max_suppression(detections, threshold=0.4):
        if len(detections) == 0:
            return np.array([])
        boxes = detections[:, :4]
        scores = detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, threshold)
        return detections[indices.flatten()] if len(indices) > 0 else np.array([])

    @staticmethod
    def draw_detections(frame, detections, class_names=None):
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = class_names[int(cls)] if class_names else f"Object {cls}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    @staticmethod
    def calculate_fps(start_time, frame_count):
        elapsed_time = time.time() - start_time.timestamp()
        return frame_count / elapsed_time if elapsed_time > 0 else 0
