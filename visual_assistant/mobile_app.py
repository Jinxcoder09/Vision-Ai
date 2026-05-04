"""
Mobile-optimized Visual Assistant using Kivy
For Android/iOS deployment
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.popup import Popup
from kivy.clock import Clock
import tempfile
import os
import threading

# Import core functionality
from core import VisualAssistant


class VisualAssistantMobile(App):
    """Mobile-optimized Visual Assistant for Android/iOS"""
    
    def build(self):
        self.title = "Visual Assistant"
        self.assistant = VisualAssistant()
        
        # State
        self.current_result = None
        self.is_processing = False
        
        # Build UI
        return self.create_main_layout()
    
    def create_main_layout(self):
        """Create the main application layout"""
        
        # Main vertical layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title_label = Label(
            text="👁️ Visual Assistant",
            size_hint_y=None,
            height=50,
            font_size='24sp',
            bold=True
        )
        main_layout.add_widget(title_label)
        
        # Camera preview
        try:
            self.camera = Camera(index=0, resolution=(640, 480), play=True)
            self.camera.size_hint_y = 0.5
            main_layout.add_widget(self.camera)
        except Exception as e:
            error_label = Label(
                text=f"Camera not available: {str(e)}",
                size_hint_y=None,
                height=200
            )
            main_layout.add_widget(error_label)
            self.camera = None
        
        # Task selection buttons
        task_layout = BoxLayout(size_hint_y=None, height=60, spacing=5)
        
        self.btn_detect = Button(text="🔍\nDetect", font_size='14sp')
        self.btn_detect.bind(on_press=lambda x: self.set_task('detection'))
        task_layout.add_widget(self.btn_detect)
        
        self.btn_navigate = Button(text="🧭\nNavigate", font_size='14sp')
        self.btn_navigate.bind(on_press=lambda x: self.set_task('navigation'))
        task_layout.add_widget(self.btn_navigate)
        
        self.btn_ocr = Button(text="📄\nRead Text", font_size='14sp')
        self.btn_ocr.bind(on_press=lambda x: self.set_task('ocr'))
        task_layout.add_widget(self.btn_ocr)
        
        self.btn_faces = Button(text="👤\nFaces", font_size='14sp')
        self.btn_faces.bind(on_press=lambda x: self.set_task('face_recognition'))
        task_layout.add_widget(self.btn_faces)
        
        main_layout.add_widget(task_layout)
        
        # Current task indicator
        self.task_label = Label(
            text="Task: Object Detection",
            size_hint_y=None,
            height=30,
            color=(0.5, 0.8, 1, 1)
        )
        main_layout.add_widget(self.task_label)
        
        # Action buttons
        action_layout = BoxLayout(size_hint_y=None, height=60, spacing=10)
        
        self.btn_capture = Button(
            text="📷 Capture & Analyze",
            font_size='18sp',
            background_color=(0, 0.5, 1, 1)
        )
        self.btn_capture.bind(on_press=self.capture_and_analyze)
        action_layout.add_widget(self.btn_capture)
        
        main_layout.add_widget(action_layout)
        
        # Speak button (initially disabled)
        self.btn_speak = Button(
            text="🔊 Speak Result",
            font_size='18sp',
            disabled=True,
            background_color=(0, 0.7, 0, 1)
        )
        self.btn_speak.bind(on_press=self.speak_result)
        main_layout.add_widget(self.btn_speak)
        
        # Results area
        result_frame = BoxLayout(
            orientation='vertical',
            size_hint_y=0.3,
            padding=10
        )
        
        result_title = Label(
            text="Results:",
            size_hint_y=None,
            height=30,
            halign='left'
        )
        result_frame.add_widget(result_title)
        
        self.result_label = Label(
            text="Capture an image to begin analysis",
            size_hint_y=None,
            height=150,
            valign='top',
            halign='center',
            text_size=(result_frame.width - 20, None),
            color=(1, 1, 1, 1)
        )
        result_frame.add_widget(self.result_label)
        
        main_layout.add_widget(result_frame)
        
        # Default task
        self.current_task = 'detection'
        self.btn_detect.background_color = (0, 0.7, 0, 1)
        
        return main_layout
    
    def set_task(self, task):
        """Set the current analysis task"""
        self.current_task = task
        
        # Reset button colors
        for btn in [self.btn_detect, self.btn_navigate, self.btn_ocr, self.btn_faces]:
            btn.background_color = (0, 0.5, 1, 1)
        
        # Highlight selected
        task_map = {
            'detection': self.btn_detect,
            'navigation': self.btn_navigate,
            'ocr': self.btn_ocr,
            'face_recognition': self.btn_faces
        }
        
        if task in task_map:
            task_map[task].background_color = (0, 0.7, 0, 1)
            
            task_names = {
                'detection': 'Object Detection',
                'navigation': 'Navigation Assist',
                'ocr': 'Text Reading (OCR)',
                'face_recognition': 'Face Recognition'
            }
            self.task_label.text = f"Task: {task_names.get(task, task)}"
    
    def capture_and_analyze(self, instance):
        """Capture image from camera and analyze"""
        if not self.camera or self.is_processing:
            return
        
        self.is_processing = True
        self.btn_capture.disabled = True
        self.result_label.text = "Capturing and analyzing..."
        
        # Run in background thread
        thread = threading.Thread(target=self._process_image)
        thread.daemon = True
        thread.start()
    
    def _process_image(self):
        """Process image in background"""
        try:
            # Capture frame
            temp_path = tempfile.mktemp(suffix='.jpg')
            self.camera.export_to_png(temp_path)
            
            # Analyze
            result = self.assistant.process_frame(
                temp_path,
                self.current_task,
                speak_result=False  # We'll speak manually
            )
            
            self.current_result = result
            
            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._update_ui(result))
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(str(e)))
    
    def _update_ui(self, result):
        """Update UI with results"""
        self.is_processing = False
        self.btn_capture.disabled = False
        
        if result['success']:
            # Display truncated result
            analysis = result['analysis']
            if len(analysis) > 300:
                analysis = analysis[:300] + "..."
            
            self.result_label.text = analysis
            self.btn_speak.disabled = False
            self.result_label.color = (0.5, 1, 0.5, 1)
        else:
            self.result_label.text = f"Error: {result.get('error', 'Unknown error')}"
            self.result_label.color = (1, 0.5, 0.5, 1)
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.is_processing = False
        self.btn_capture.disabled = False
        self.result_label.text = f"Error: {error_msg}"
        self.result_label.color = (1, 0.5, 0.5, 1)
    
    def speak_result(self, instance):
        """Speak the analysis result"""
        if not self.current_result or not self.current_result['success']:
            return
        
        self.btn_speak.disabled = True
        self.result_label.text = "Speaking..."
        
        # Generate speech in background
        thread = threading.Thread(target=self._generate_and_play_speech)
        thread.daemon = True
        thread.start()
    
    def _generate_and_play_speech(self):
        """Generate speech and play audio"""
        try:
            if self.current_result.get('audio_file'):
                # Audio already generated, just play it
                self._play_audio(self.current_result['audio_file'])
            else:
                # Generate new audio
                analysis = self.current_result['analysis']
                
                timestamp = tempfile.mktemp(suffix='.wav')
                success = self.assistant.tts_service.text_to_speech(
                    analysis,
                    timestamp
                )
                
                if success:
                    self._play_audio(timestamp)
            
            Clock.schedule_once(lambda dt: setattr(self, 'btn_speak.disabled', False))
            Clock.schedule_once(lambda dt: setattr(
                self.result_label, 'text',
                self.current_result['analysis'][:300] + "..."
            ))
            
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(f"Speech error: {str(e)}"))
    
    def _play_audio(self, audio_path):
        """Play audio file (platform-specific implementation needed)"""
        try:
            # For Android, use android.media or plyer
            # For iOS, use AVFoundation
            # This is a placeholder - actual implementation depends on platform
            
            import subprocess
            
            # Try playing with system player
            if os.path.exists(audio_path):
                # Linux/Android
                subprocess.call(['aplay', audio_path], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Audio playback error: {e}")


if __name__ == '__main__':
    VisualAssistantMobile().run()
