import cv2
import numpy as np
import json
import os
from datetime import datetime

class DetectionUtils:
    @staticmethod
    def load_config(config_path):
        """
        Load and validate configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return None

    @staticmethod
    def preprocess_frame(frame, target_size=(224, 224)):
        """
        Preprocess a frame for detection
        """
        try:
            # Resize frame
            resized = cv2.resize(frame, target_size)
            
            # Convert to RGB if necessary
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
            elif resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            print(f"Error preprocessing frame: {str(e)}")
            return None

    @staticmethod
    def draw_detections(frame, detections, classes=None):
        """
        Draw bounding boxes and labels on frame
        """
        try:
            frame_copy = frame.copy()
            
            for detection in detections:
                if len(detection) >= 5:  # x, y, w, h, confidence, [class_id]
                    x, y, w, h = map(int, detection[:4])
                    confidence = detection[4]
                    class_id = int(detection[5]) if len(detection) > 5 else 0
                    
                    # Draw bounding box
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Prepare label text
                    label = f"{classes[class_id] if classes else 'Object'}: {confidence:.2f}"
                    
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame_copy, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame_copy, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return frame_copy
        except Exception as e:
            print(f"Error drawing detections: {str(e)}")
            return frame

    @staticmethod
    def save_detection(frame, detection, save_dir="data/detections"):
        """
        Save detection results and frame
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"detection_{timestamp}"
            
            # Save frame
            frame_path = os.path.join(save_dir, f"{base_filename}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Save detection data
            data_path = os.path.join(save_dir, f"{base_filename}.json")
            with open(data_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'detection': detection.tolist() if isinstance(detection, np.ndarray) else detection
                }, f)
            
            return True
        except Exception as e:
            print(f"Error saving detection: {str(e)}")
            return False

    @staticmethod
    def calculate_fps(start_time, frame_count):
        """
        Calculate frames per second
        """
        try:
            current_time = datetime.now()
            time_diff = (current_time - start_time).total_seconds()
            fps = frame_count / time_diff if time_diff > 0 else 0
            return fps
        except Exception as e:
            print(f"Error calculating FPS: {str(e)}")
            return 0

    @staticmethod
    def non_max_suppression(boxes, scores, threshold=0.3):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        try:
            if len(boxes) == 0:
                return []
            
            # Convert to float
            boxes = boxes.astype(float)
            
            # Get coordinates
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 0] + boxes[:, 2]
            y2 = boxes[:, 1] + boxes[:, 3]
            
            # Calculate area
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            # Sort by confidence
            idxs = np.argsort(scores)
            
            pick = []
            while len(idxs) > 0:
                # Get index of highest scoring box
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)
                
                # Find overlapping boxes
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])
                
                # Calculate overlap area
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                overlap = (w * h) / area[idxs[:last]]
                
                # Remove overlapping boxes
                idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > threshold)[0])))
            
            return boxes[pick].astype("int")
        except Exception as e:
            print(f"Error in non-max suppression: {str(e)}")
            return []

    @staticmethod
    def validate_detection(detection, min_size=20, min_confidence=0.5):
        """
        Validate detection results
        """
        try:
            if len(detection) < 5:  # Must have at least x, y, w, h, confidence
                return False
                
            x, y, w, h, confidence = detection[:5]
            
            # Check size
            if w < min_size or h < min_size:
                return False
                
            # Check confidence
            if confidence < min_confidence:
                return False
                
            # Check coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False
                
            return True
        except Exception as e:
            print(f"Error validating detection: {str(e)}")
            return False
