import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import json
import os
class ObjectDetectionModel:
    def __init__(self, config_path='../config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Builds a simple CNN model for object detection.
        This is a basic implementation that will be enhanced through online learning.
        """
        model = tf.keras.Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.config['classes']) or 1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, image):
        """
        Perform object detection on an image
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(processed_image)
        
        return self._process_predictions(predictions)
    
    def _preprocess_image(self, image):
        """
        Preprocess image for model input
        """
        # Resize image to expected input size
        image = tf.image.resize(image, (224, 224))
        # Expand dimensions to create batch
        image = tf.expand_dims(image, 0)
        # Normalize pixel values
        image = image / 255.0
        
        return image
    
    def _process_predictions(self, predictions):
        """
        Process model predictions into a more usable format
        """
        # Filter predictions based on confidence threshold
        confident_predictions = predictions[predictions >= self.config['confidence_threshold']]
        
        return confident_predictions
    
    def update(self, new_weights):
        """
        Update model weights with online learning data
        """
        try:
            self.model.set_weights(new_weights)
            return True
        except Exception as e:
            print(f"Error updating model weights: {str(e)}")
            return False
    
    def save(self):
        """
        Save the current model state
        """
        self.model.save(self.config['model_save_path'])
    
    def load(self):
        """
        Load a saved model state
        """
        if os.path.exists(self.config['model_save_path']):
            self.model = tf.keras.models.load_model(self.config['model_save_path'])
            return True
        return False
