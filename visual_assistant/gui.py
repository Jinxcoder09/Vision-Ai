"""
Visual Assistant - Windows GUI Application
A user-friendly interface for visually impaired users
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from pathlib import Path
from datetime import datetime
import wave
import pygame

# Import core functionality
from core import VisualAssistant


class VisualAssistantGUI:
    """Main GUI Application for Visual Assistant"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Assistant for Visually Impaired")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        
        # Initialize core assistant
        self.assistant = VisualAssistant()
        
        # Initialize audio mixer
        pygame.mixer.init()
        
        # Current state
        self.current_image_path = None
        self.current_task = "detection"
        self.is_processing = False
        self.camera_running = False
        
        # Setup UI
        self.setup_ui()
        
        # Apply accessibility settings
        self.apply_accessibility_settings()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#1a1a2e'
        card_bg = '#16213e'
        text_color = '#eaeaea'
        accent_color = '#0f3460'
        highlight_color = '#e94560'
        
        # Main container
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title Section
        title_frame = tk.Frame(main_frame, bg=bg_color)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            title_frame,
            text="👁️ Visual Assistant",
            font=("Arial", 28, "bold"),
            bg=bg_color,
            fg=text_color
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(
            title_frame,
            text="AI-Powered Assistance for the Visually Impaired",
            font=("Arial", 12),
            bg=bg_color,
            fg='#a0a0a0'
        )
        subtitle_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Task Selection Frame
        task_frame = tk.LabelFrame(
            main_frame,
            text="Select Task",
            font=("Arial", 14, "bold"),
            bg=card_bg,
            fg=text_color,
            padx=15,
            pady=15
        )
        task_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.task_var = tk.StringVar(value="detection")
        
        tasks = [
            ("🔍 Object Detection", "detection"),
            ("🧭 Navigation Assist", "navigation"),
            ("📄 Read Text (OCR)", "ocr"),
            ("👤 Face Recognition", "face_recognition")
        ]
        
        for i, (text, value) in enumerate(tasks):
            rb = tk.Radiobutton(
                task_frame,
                text=text,
                variable=self.task_var,
                value=value,
                command=self.on_task_change,
                font=("Arial", 12),
                bg=card_bg,
                fg=text_color,
                selectcolor=accent_color,
                activebackground=card_bg,
                activeforeground=text_color
            )
            rb.grid(row=0, column=i, padx=10, sticky='w')
        
        task_frame.columnconfigure(0, weight=1)
        task_frame.columnconfigure(1, weight=1)
        task_frame.columnconfigure(2, weight=1)
        task_frame.columnconfigure(3, weight=1)
        
        # Image Input Section
        input_frame = tk.LabelFrame(
            main_frame,
            text="Image Input",
            font=("Arial", 14, "bold"),
            bg=card_bg,
            fg=text_color,
            padx=15,
            pady=15
        )
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Buttons for image input
        btn_frame1 = tk.Frame(input_frame, bg=card_bg)
        btn_frame1.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_load_image = tk.Button(
            btn_frame1,
            text="📁 Load Image",
            command=self.load_image,
            font=("Arial", 12, "bold"),
            bg=accent_color,
            fg=text_color,
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_load_image.pack(side=tk.LEFT, padx=5)
        
        self.btn_use_camera = tk.Button(
            btn_frame1,
            text="📷 Use Camera",
            command=self.use_camera,
            font=("Arial", 12, "bold"),
            bg=accent_color,
            fg=text_color,
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_use_camera.pack(side=tk.LEFT, padx=5)
        
        # Image preview label
        self.image_preview = tk.Label(
            input_frame,
            text="No image loaded",
            font=("Arial", 11),
            bg=bg_color,
            fg='#808080',
            height=3
        )
        self.image_preview.pack(fill=tk.X, pady=(10, 0))
        
        # Process Button
        self.btn_process = tk.Button(
            main_frame,
            text="▶️ Analyze & Speak",
            command=self.process_image,
            font=("Arial", 16, "bold"),
            bg=highlight_color,
            fg=text_color,
            padx=40,
            pady=15,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_process.pack(pady=20)
        
        # Voice Settings
        voice_frame = tk.LabelFrame(
            main_frame,
            text="Voice Settings",
            font=("Arial", 14, "bold"),
            bg=card_bg,
            fg=text_color,
            padx=15,
            pady=15
        )
        voice_frame.pack(fill=tk.X, pady=(0, 15))
        
        voice_inner = tk.Frame(voice_frame, bg=card_bg)
        voice_inner.pack(fill=tk.X)
        
        tk.Label(
            voice_inner,
            text="Voice:",
            font=("Arial", 12),
            bg=card_bg,
            fg=text_color
        ).grid(row=0, column=0, padx=(0, 10))
        
        self.voice_combo = ttk.Combobox(
            voice_inner,
            values=[v["name"] for v in self.assistant.tts_service.get_available_voices()],
            state="readonly",
            font=("Arial", 11),
            width=20
        )
        self.voice_combo.set("Default")
        self.voice_combo.grid(row=0, column=1, padx=5)
        self.voice_combo.bind('<<ComboboxSelected>>', self.on_voice_change)
        
        self.auto_expression_var = tk.BooleanVar(value=True)
        auto_expr_cb = tk.Checkbutton(
            voice_inner,
            text="Auto Expressions (Recommended)",
            variable=self.auto_expression_var,
            font=("Arial", 11),
            bg=card_bg,
            fg=text_color,
            selectcolor=card_bg
        )
        auto_expr_cb.grid(row=0, column=2, padx=20)
        
        # Results Section
        results_frame = tk.LabelFrame(
            main_frame,
            text="Analysis Results",
            font=("Arial", 14, "bold"),
            bg=card_bg,
            fg=text_color,
            padx=15,
            pady=15
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text output
        self.result_text = tk.Text(
            results_frame,
            font=("Consolas", 11),
            bg=bg_color,
            fg=text_color,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            relief=tk.FLAT,
            insertbackground=text_color
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.result_text, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # Status Bar
        status_frame = tk.Frame(main_frame, bg=accent_color)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready - Load an image to begin",
            font=("Arial", 11),
            bg=accent_color,
            fg=text_color,
            padx=10,
            pady=8,
            anchor='w'
        )
        self.status_label.pack(fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def apply_accessibility_settings(self):
        """Apply accessibility enhancements"""
        # High contrast mode
        self.root.option_add('*Font', 'Arial 11')
        
        # Keyboard navigation
        self.root.bind('<Alt+L>', lambda e: self.load_image())
        self.root.bind('<Alt+A>', lambda e: self.process_image())
        self.root.bind('<F5>', lambda e: self.process_image())
        
        # Screen reader friendly
        self.root.wm_attributes('-topmost', False)
    
    def on_task_change(self):
        """Handle task selection change"""
        self.current_task = self.task_var.get()
        task_names = {
            "detection": "Object Detection",
            "navigation": "Navigation Assist",
            "ocr": "Read Text (OCR)",
            "face_recognition": "Face Recognition"
        }
        self.update_status(f"Selected: {task_names.get(self.current_task, 'Unknown')}")
    
    def on_voice_change(self, event):
        """Handle voice selection change"""
        voice_name = self.voice_combo.get()
        voices = self.assistant.tts_service.get_available_voices()
        voice_id = next((v["id"] for v in voices if v["name"] == voice_name), "default")
        self.assistant.tts_service.set_voice(voice_id)
    
    def load_image(self):
        """Load image from file"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_image_path = filepath
            filename = os.path.basename(filepath)
            self.image_preview.config(
                text=f"✓ Loaded: {filename}",
                fg='#4caf50'
            )
            self.update_status(f"Image loaded: {filename}")
    
    def use_camera(self):
        """Use camera for live capture"""
        messagebox.showinfo(
            "Camera Mode",
            "Camera feature will capture a photo from your webcam.\n\n"
            "Make sure you have a webcam connected."
        )
        
        try:
            import cv2
            
            # Open camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                return
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Save temporary image
                temp_path = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(temp_path, frame)
                
                self.current_image_path = temp_path
                self.image_preview.config(
                    text=f"✓ Captured from camera",
                    fg='#4caf50'
                )
                self.update_status("Image captured from camera")
            else:
                messagebox.showerror("Error", "Failed to capture image")
                
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")
    
    def process_image(self):
        """Process the loaded image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "Image file not found")
            return
        
        if self.is_processing:
            return
        
        # Start processing
        self.is_processing = True
        self.btn_process.config(state='disabled', text='⏳ Processing...')
        self.progress.start()
        self.update_status("Analyzing image...")
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Processing...\n")
        
        # Process in background thread
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True
        thread.start()
    
    def _process_thread(self):
        """Background processing thread"""
        try:
            result = self.assistant.process_frame(
                self.current_image_path,
                self.current_task,
                speak_result=True
            )
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_results(result))
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _update_results(self, result):
        """Update UI with results"""
        self.is_processing = False
        self.btn_process.config(state='normal', text='▶️ Analyze & Speak')
        self.progress.stop()
        
        if result["success"]:
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=" * 50 + "\n")
            self.result_text.insert(tk.END, f"Task: {result['task'].upper()}\n")
            self.result_text.insert(tk.END, f"Time: {result['timestamp']}\n")
            self.result_text.insert(tk.END, "=" * 50 + "\n\n")
            self.result_text.insert(tk.END, result["analysis"])
            
            # Update status
            self.update_status("Analysis complete!")
            
            # Play audio if available
            if result.get("audio_file"):
                self.play_audio(result["audio_file"])
                self.update_status("Analysis complete - Audio playing...")
        else:
            self._handle_error(result.get("error", "Unknown error"))
    
    def _handle_error(self, error_msg):
        """Handle processing error"""
        self.is_processing = False
        self.btn_process.config(state='normal', text='▶️ Analyze & Speak')
        self.progress.stop()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Error: {error_msg}\n")
        self.update_status("Processing failed")
        messagebox.showerror("Error", error_msg)
    
    def play_audio(self, audio_file):
        """Play audio file"""
        try:
            if os.path.exists(audio_file):
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = VisualAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
