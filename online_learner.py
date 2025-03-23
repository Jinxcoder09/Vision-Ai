import requests
import json
import numpy as np
import time
import threading
from datetime import datetime
import os

class OnlineLearner:
    def __init__(self, config_path='../config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.last_update = datetime.now()
        self.is_running = False
        self.temp_data_path = "data/temp/"
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_data_path, exist_ok=True)
    
    def start(self, model):
        """
        Start the online learning process in a separate thread
        """
        self.is_running = True
        self.model = model
        
        # Start the learning thread
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
    
    def stop(self):
        """
        Stop the online learning process
        """
        self.is_running = False
        if hasattr(self, 'learning_thread'):
            self.learning_thread.join()
    
    def _learning_loop(self):
        """
        Main loop for continuous online learning
        """
        while self.is_running:
            try:
                # Check if it's time for an update
                current_time = datetime.now()
                time_diff = (current_time - self.last_update).total_seconds()
                
                if time_diff >= self.config['online_learning']['update_interval']:
                    self._perform_update()
                    self.last_update = current_time
                
                # Sleep for a short period to prevent excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in learning loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _perform_update(self):
        """
        Perform a single update cycle
        """
        try:
            # Fetch new training data from API
            new_data = self._fetch_training_data()
            
            if new_data:
                # Process and validate the new data
                processed_data = self._process_training_data(new_data)
                
                if processed_data:
                    # Update the model
                    self.model.update(processed_data)
                    
                    # Save temporary data for backup
                    self._save_temp_data(processed_data)
        
        except Exception as e:
            print(f"Error during update: {str(e)}")
    
    def _fetch_training_data(self):
        """
        Fetch new training data from the API
        """
        try:
            response = requests.get(
                self.config['api_endpoint'],
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed with status code: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching training data: {str(e)}")
            return None
    
    def _process_training_data(self, data):
        """
        Process and validate the received training data
        """
        try:
            # Validate data format
            if not self._validate_data_format(data):
                return None
            
            # Convert data to appropriate format for model update
            processed_data = self._convert_data_format(data)
            
            # Filter based on confidence threshold
            processed_data = self._filter_by_confidence(processed_data)
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing training data: {str(e)}")
            return None
    
    def _validate_data_format(self, data):
        """
        Validate the format of received data
        """
        required_fields = ['weights', 'metadata']
        return all(field in data for field in required_fields)
    
    def _convert_data_format(self, data):
        """
        Convert API data format to model format
        """
        return np.array(data['weights'])
    
    def _filter_by_confidence(self, data):
        """
        Filter data based on confidence threshold
        """
        confidence_threshold = self.config['online_learning']['min_confidence']
        return data[data >= confidence_threshold]
    
    def _save_temp_data(self, data):
        """
        Save temporary data for backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.temp_data_path}backup_{timestamp}.npz"
        
        try:
            np.savez_compressed(filename, data=data)
        except Exception as e:
            print(f"Error saving temporary data: {str(e)}")
    
    def get_learning_status(self):
        """
        Get the current status of the online learning process
        """
        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat(),
            'temp_data_size': self._get_temp_data_size()
        }
    
    def _get_temp_data_size(self):
        """
        Get the size of temporary data storage
        """
        total_size = 0
        for filename in os.listdir(self.temp_data_path):
            filepath = os.path.join(self.temp_data_path, filename)
            total_size += os.path.getsize(filepath)
        return total_size
