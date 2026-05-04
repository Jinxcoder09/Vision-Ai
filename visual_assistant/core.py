"""
Visual Assistant for Visually Impaired
A comprehensive AI-powered assistance system using NVIDIA models
"""

import requests
import base64
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import queue

# API Configuration
class APIConfig:
    # Llama 3.2 Vision (Object Detection & Navigation)
    LLAMA_API_KEY = "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"
    LLAMA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    LLAMA_MODEL = "meta/llama-3.2-90b-vision-instruct"
    
    # TTS (Audio Output)
    TTS_API_KEY = "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"
    TTS_URL = "https://integrate.api.nvidia.com/v1/audio/speech"
    TTS_MODEL = "nvidia/magpie-tts-multilingual"
    
    # OCR (Text Extraction)
    OCR_API_KEY = "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"
    OCR_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    OCR_MODEL = "nvidia/nemotron-ocr-v1"
    
    # LLM for Text Processing
    LLM_API_KEY = "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"
    LLM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    LLM_MODEL = "minimaxai/minimax-m2.7"


class ImageEncoder:
    """Helper class to encode images to base64"""
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def encode_image_from_bytes(image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')


class LlamaVisionService:
    """Object Detection, Navigation, and Face Recognition using Llama 3.2 90B"""
    
    def __init__(self):
        self.api_key = APIConfig.LLAMA_API_KEY
        self.url = APIConfig.LLAMA_URL
        self.model = APIConfig.LLAMA_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_frame(self, image_base64: str, task_type: str = "detection") -> str:
        """
        Analyze frame for different tasks
        task_type: 'detection', 'navigation', 'face_recognition'
        """
        
        prompts = {
            "detection": """You are assisting a visually impaired person. Analyze this image and provide:
1. List all objects detected with their positions (left, right, center, near, far)
2. Describe the scene briefly but comprehensively
3. Highlight any potential obstacles or hazards
4. Mention any people present and their approximate positions
Be concise but informative. Speak naturally as if describing to someone who cannot see.""",
            
            "navigation": """You are a navigation assistant for a visually impaired person. Analyze this image and provide:
1. Describe the path ahead and any obstacles
2. Identify walkable areas and potential dangers
3. Mention stairs, doors, elevators, or changes in elevation
4. Provide clear directional guidance (turn left/right, go straight, stop)
5. Note any traffic, vehicles, or moving objects
Be very clear and direct with instructions. Safety is the priority.""",
            
            "face_recognition": """You are helping a visually impaired person recognize faces. Analyze this image and:
1. Count how many people/faces are visible
2. Describe each person's approximate age, gender, and position
3. Note any distinctive features (glasses, beard, hair color, clothing)
4. Describe facial expressions and emotions if visible
5. Estimate distances to each person
Be respectful and descriptive without making assumptions."""
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts.get(task_type, prompts["detection"])
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Unable to analyze the image. Please try again."
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to vision service: {str(e)}"
        except Exception as e:
            return f"Analysis error: {str(e)}"


class OCRService:
    """OCR Service using NVIDIA Nemotron OCR"""
    
    def __init__(self):
        self.api_key = APIConfig.OCR_API_KEY
        self.url = APIConfig.OCR_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def extract_text(self, image_base64: str) -> str:
        """Extract text from image using OCR"""
        
        prompt = """Extract all text from this image. Preserve the structure and order of the text.
Include headings, paragraphs, lists, and any other text elements.
If there are multiple languages, identify them.
Return only the extracted text without additional commentary."""
        
        payload = {
            "model": "nvidia/nemotron-ocr-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "No text detected in the image."
                
        except requests.exceptions.RequestException as e:
            return f"OCR error: {str(e)}"


class TextProcessor:
    """LLM service to process and clean OCR text"""
    
    def __init__(self):
        self.api_key = APIConfig.LLM_API_KEY
        self.url = APIConfig.LLM_URL
        self.model = APIConfig.LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def clean_and_normalize(self, ocr_text: str) -> str:
        """Clean OCR text, remove noise, and make it natural for speech"""
        
        prompt = f"""You are helping a visually impaired person by cleaning up OCR text for audio playback.
Process the following text and:
1. Remove any OCR errors, artifacts, or gibberish
2. Fix obvious typos and formatting issues
3. Remove unnecessary symbols, page numbers, or headers/footers if they seem irrelevant
4. Make the text flow naturally for speech
5. Keep all meaningful content intact
6. If the text is a sign, label, or important notice, preserve it exactly
7. Remove repeated words caused by OCR errors

Here is the OCR text to process:
{ocr_text}

Return only the cleaned text, ready to be spoken aloud."""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return ocr_text  # Return original if processing fails
                
        except requests.exceptions.RequestException as e:
            print(f"Text processing error: {e}")
            return ocr_text  # Return original on error


class TTSService:
    """Text-to-Speech Service using NVIDIA Magpie TTS Multilingual"""
    
    def __init__(self):
        self.api_key = APIConfig.TTS_API_KEY
        self.url = APIConfig.TTS_URL
        self.model = APIConfig.TTS_MODEL
        self.current_voice = "default"
        self.available_voices = [
            {"id": "default", "name": "Default", "description": "Natural balanced voice"},
            {"id": "female_calm", "name": "Calm Female", "description": "Soothing female voice"},
            {"id": "male_clear", "name": "Clear Male", "description": "Clear male voice"},
            {"id": "expressive", "name": "Expressive", "description": "Emotionally expressive voice"}
        ]
    
    def set_voice(self, voice_id: str):
        """Set the voice for TTS"""
        valid_voices = [v["id"] for v in self.available_voices]
        if voice_id in valid_voices:
            self.current_voice = voice_id
            return True
        return False
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        return self.available_voices
    
    def text_to_speech(self, text: str, output_path: str = "output_audio.wav", 
                       auto_expressions: bool = True) -> bool:
        """
        Convert text to speech and save to file
        auto_expressions: Automatically adjust tone based on content
        """
        
        # Determine appropriate voice and style based on content
        voice_params = self._determine_voice_and_style(text, auto_expressions)
        
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice_params["voice"],
            "language_code": voice_params.get("language", "en-US"),
            "speed": voice_params.get("speed", 1.0),
            "pitch": voice_params.get("pitch", 0)
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Save audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"TTS error: {e}")
            return False
        except Exception as e:
            print(f"Audio save error: {e}")
            return False
    
    def _determine_voice_and_style(self, text: str, auto_expressions: bool) -> dict:
        """Automatically determine voice parameters based on text content"""
        
        if not auto_expressions:
            return {"voice": self.current_voice}
        
        text_lower = text.lower()
        
        # Emergency/warning detection
        if any(word in text_lower for word in ["warning", "danger", "caution", "stop", "careful", "obstacle"]):
            return {
                "voice": "expressive",
                "speed": 1.1,
                "pitch": 5,
                "language": "en-US"
            }
        
        # Navigation instructions
        elif any(word in text_lower for word in ["turn", "go", "walk", "step", "left", "right", "straight"]):
            return {
                "voice": "male_clear" if self.current_voice == "default" else self.current_voice,
                "speed": 0.95,
                "pitch": 0,
                "language": "en-US"
            }
        
        # Reading text/documents
        elif any(word in text_lower for word in ["document", "sign", "text", "letter", "page"]):
            return {
                "voice": "female_calm" if self.current_voice == "default" else self.current_voice,
                "speed": 0.9,
                "pitch": -2,
                "language": "en-US"
            }
        
        # Friendly/descriptive content
        else:
            return {
                "voice": self.current_voice,
                "speed": 1.0,
                "pitch": 0,
                "language": "en-US"
            }
    
    def stream_audio(self, text: str):
        """Stream audio directly (for real-time feedback)"""
        # This would require audio playback library
        # For now, save to file and let the GUI handle playback
        output_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        return self.text_to_speech(text, output_path)


class VisualAssistant:
    """Main controller for the Visual Assistant system"""
    
    def __init__(self):
        self.vision_service = LlamaVisionService()
        self.ocr_service = OCRService()
        self.text_processor = TextProcessor()
        self.tts_service = TTSService()
        self.image_encoder = ImageEncoder()
        
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.current_task = None
        
        # Voice profiles
        self.voice_profiles = {
            "navigation": "male_clear",
            "object_detection": "default",
            "reading": "female_calm",
            "face_recognition": "expressive"
        }
    
    def process_frame(self, image_path: str, task_type: str = "detection", 
                     speak_result: bool = True) -> Dict[str, Any]:
        """
        Process a single frame/image
        task_type: 'detection', 'navigation', 'ocr', 'face_recognition'
        """
        
        result = {
            "success": False,
            "task": task_type,
            "timestamp": datetime.now().isoformat(),
            "analysis": "",
            "audio_file": None,
            "error": None
        }
        
        try:
            # Encode image
            image_base64 = self.image_encoder.encode_image(image_path)
            
            # Process based on task type
            if task_type == "ocr":
                # OCR + Text Processing pipeline
                raw_text = self.ocr_service.extract_text(image_base64)
                cleaned_text = self.text_processor.clean_and_normalize(raw_text)
                result["analysis"] = cleaned_text
                
            elif task_type in ["detection", "navigation", "face_recognition"]:
                # Vision analysis
                analysis = self.vision_service.analyze_frame(image_base64, task_type)
                result["analysis"] = analysis
            
            else:
                result["error"] = f"Unknown task type: {task_type}"
                return result
            
            # Generate audio if requested
            if speak_result and result["analysis"]:
                # Set appropriate voice for task
                voice_profile = self.voice_profiles.get(task_type, "default")
                self.tts_service.set_voice(voice_profile)
                
                # Generate speech with auto-expressions
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                audio_file = f"audio_{task_type}_{timestamp}.wav"
                
                if self.tts_service.text_to_speech(result["analysis"], audio_file):
                    result["audio_file"] = audio_file
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        return result
    
    def process_live_stream(self, image_source, task_type: str = "detection",
                           callback=None):
        """
        Process live video stream (for real-time assistance)
        image_source: camera index or video file path
        """
        import cv2
        
        # Open camera or video file
        if isinstance(image_source, int):
            cap = cv2.VideoCapture(image_source)
        else:
            cap = cv2.VideoCapture(image_source)
        
        if not cap.isOpened():
            raise Exception("Cannot open video source")
        
        frame_count = 0
        last_process_time = 0
        process_interval = 2.0  # Process every 2 seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Process frame at intervals
                if current_time - last_process_time >= process_interval:
                    # Save frame temporarily
                    temp_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # Process in background thread
                    threading.Thread(
                        target=self._process_and_callback,
                        args=(temp_path, task_type, callback),
                        daemon=True
                    ).start()
                    
                    last_process_time = current_time
                    frame_count += 1
                
                # Check for stop signal
                if not self.is_processing:
                    break
                    
        finally:
            cap.release()
    
    def _process_and_callback(self, image_path: str, task_type: str, callback):
        """Process frame and call callback with result"""
        result = self.process_frame(image_path, task_type, speak_result=True)
        
        if callback:
            callback(result)
        
        # Clean up temp file
        try:
            Path(image_path).unlink()
        except:
            pass
    
    def start_listening(self):
        """Start audio processing thread"""
        self.is_processing = True
    
    def stop_listening(self):
        """Stop audio processing"""
        self.is_processing = False
    
    def set_voice_profile(self, task_type: str, voice_id: str):
        """Set voice profile for specific task type"""
        if task_type in self.voice_profiles:
            self.voice_profiles[task_type] = voice_id


# Utility functions
def create_demo_sequence():
    """Create a demo sequence showing all capabilities"""
    assistant = VisualAssistant()
    
    print("=" * 60)
    print("VISUAL ASSISTANT DEMO")
    print("=" * 60)
    
    # Example usage (requires actual image files)
    # Uncomment and provide actual image paths to test
    
    # Object Detection
    # result = assistant.process_frame("path/to/image.jpg", "detection")
    # print(f"Detection: {result['analysis']}")
    
    # OCR
    # result = assistant.process_frame("path/to/text_image.jpg", "ocr")
    # print(f"OCR Text: {result['analysis']}")
    
    # Navigation
    # result = assistant.process_frame("path/to/scene.jpg", "navigation")
    # print(f"Navigation: {result['analysis']}")
    
    # Face Recognition
    # result = assistant.process_frame("path/to/people.jpg", "face_recognition")
    # print(f"Faces: {result['analysis']}")
    
    print("\nDemo complete!")
    print("Provide image paths to test actual functionality.")


if __name__ == "__main__":
    create_demo_sequence()
