import cv2
import numpy as np
from datetime import datetime
import argparse
import sys
import signal
from detector import ObjectDetectionModel, OnlineLearner, DetectionUtils

class ObjectDetectionSystem:
    def __init__(self, config_path='config.json'):
        self.config = DetectionUtils.load_config(config_path)
        if not self.config:
            raise ValueError("Failed to load configuration")

        # Initialize components
        self.model = ObjectDetectionModel(config_path)
        self.online_learner = OnlineLearner(config_path)
        
        # Initialize video capture
        self.cap = None
        self.running = False
        
        # Statistics
        self.frame_count = 10
        self.start_time = None
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def initialize_capture(self, source=0):
        """
        Initialize video capture from camera or video file
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
    
    def start(self):
        """
        Start the object detection system
        """
        try:
            # Initialize video capture if not already initialized
            if not self.cap:
                self.initialize_capture()
            
            # Start online learning in background
            self.online_learner.start(self.model)
            
            # Start processing frames
            self.running = True
            self.start_time = datetime.now()
            self.process_frames()
            
        except Exception as e:
            print(f"Error starting system: {str(e)}")
            self.cleanup()
    
    def process_frames(self):
        """
        Main loop for processing video frames
        """
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Preprocess frame
                processed_frame = DetectionUtils.preprocess_frame(frame)
                if processed_frame is None:
                    continue
                
                # Perform detection
                detections = self.model.predict(processed_frame)
                
                # Apply non-max suppression
                if len(detections) > 0:
                    detections = DetectionUtils.non_max_suppression(
                        detections,
                        detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
                    )
                
                # Draw detections
                frame_with_detections = DetectionUtils.draw_detections(
                    frame,
                    detections,
                    self.config.get('classes', None)
                )
                
                # Calculate and display FPS
                self.frame_count += 1
                fps = DetectionUtils.calculate_fps(self.start_time, self.frame_count)
                cv2.putText(frame_with_detections, f"FPS: {fps:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Object Detection', frame_with_detections)
                
                # Save detection if configured
                if self.config.get('save_detections', False) and len(detections) > 0:
                    DetectionUtils.save_detection(frame_with_detections, detections)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.running = False
        
        # Stop online learning
        if hasattr(self, 'online_learner'):
            self.online_learner.stop()
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        # Close all windows
        cv2.destroyAllWindows()
    
    def signal_handler(self, signum, frame):
        """
        Handle shutdown signals
        """
        print("\nShutting down...")
        self.cleanup()
        sys.exit(0)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Live Learning Object Detection System')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, or path to video file)')
    return parser.parse_args()

def main():
    """
    Main entry point
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create and start system
        system = ObjectDetectionSystem(args.config)
        
        # Initialize video source
        source = 0 if args.source == '0' else args.source
        system.initialize_capture(source)
        
        # Start processing
        system.start()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'system' in locals():
            system.cleanup()

if __name__ == "__main__":
    main()
