# Android APK Build Guide

This guide explains how to build an Android APK version of the Visual Assistant.

## Prerequisites

1. **Install Python 3.8+**
2. **Install Java JDK 11**
   - Download from: https://www.oracle.com/java/technologies/javase/jdk11-archive-downloads.html
   - Set JAVA_HOME environment variable

3. **Install Android SDK**
   - Download from: https://developer.android.com/studio
   - Install command-line tools
   - Set ANDROID_HOME environment variable

4. **Install Buildozer** (for Kivy-based APK)
   ```bash
   pip install buildozer
   ```

5. **Install Cython and other dependencies**
   ```bash
   pip install cython pillow requests
   ```

## Option 1: Using Kivy (Recommended for Mobile)

### Step 1: Create Kivy App

Create a new file `mobile_app.py`:

```python
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from core import VisualAssistant
import tempfile
import os

class VisualAssistantMobile(App):
    def build(self):
        self.assistant = VisualAssistant()
        
        layout = BoxLayout(orientation='vertical')
        
        # Camera preview
        self.camera = Camera(index=0, resolution=(640, 480))
        layout.add_widget(self.camera)
        
        # Control buttons
        btn_layout = BoxLayout(size_hint_y=None, height=100)
        
        btn_capture = Button(text='📷 Capture')
        btn_capture.bind(on_press=self.capture_and_analyze)
        btn_layout.add_widget(btn_capture)
        
        self.btn_speak = Button(text='🔊 Speak Result', disabled=True)
        self.btn_speak.bind(on_press=self.speak_result)
        btn_layout.add_widget(self.btn_speak)
        
        layout.add_widget(btn_layout)
        
        # Result label
        self.result_label = Label(text='Capture an image to begin', 
                                  size_hint_y=None, height=200)
        layout.add_widget(self.result_label)
        
        self.current_result = None
        
        return layout
    
    def capture_and_analyze(self, instance):
        # Capture frame from camera
        temp_path = tempfile.mktemp(suffix='.jpg')
        self.camera.export_to_png(temp_path)
        
        # Analyze
        self.result_label.text = 'Analyzing...'
        result = self.assistant.process_frame(temp_path, 'detection')
        
        self.current_result = result
        self.result_label.text = result['analysis']
        self.btn_speak.disabled = False
        
        # Cleanup
        os.unlink(temp_path)
    
    def speak_result(self, instance):
        if self.current_result and self.current_result.get('audio_file'):
            # Play audio (Android-specific implementation needed)
            pass

if __name__ == '__main__':
    VisualAssistantMobile().run()
```

### Step 2: Create buildozer.spec

Run once to generate spec file:
```bash
buildozer init
```

Edit `buildozer.spec`:

```ini
[app]
title = Visual Assistant
package.name = visualassistant
package.domain = org.yourname

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0.0

requirements = python3,kivy,requests,opencv,pillow
orientation = portrait

android.permissions = CAMERA,INTERNET,RECORD_AUDIO,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.ndk = 23b
android.arch = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
```

### Step 3: Build APK

```bash
# Debug build
buildozer -v android debug

# Release build
buildozer -v android release
```

The APK will be in `bin/` folder.

## Option 2: Using BeeWare/Briefcase

### Step 1: Install Briefcase

```bash
pip install briefcase
```

### Step 2: Create New Project

```bash
briefcase new
```

Follow prompts to create project structure.

### Step 3: Add Your Code

Replace the generated app code with Visual Assistant logic.

### Step 4: Build for Android

```bash
briefcase create android
briefcase build android
briefcase run android
```

## Option 3: Using React Native + Python Backend

For a more native experience:

1. Create React Native frontend
2. Run Python backend as a service
3. Communicate via HTTP API

### Backend API Server

Create `api_server.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from core import VisualAssistant
import base64
import tempfile

app = Flask(__name__)
CORS(app)
assistant = VisualAssistant()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_b64 = data['image']
    task = data.get('task', 'detection')
    
    # Decode and save temporarily
    img_data = base64.b64decode(image_b64)
    temp_path = tempfile.mktemp(suffix='.jpg')
    
    with open(temp_path, 'wb') as f:
        f.write(img_data)
    
    result = assistant.process_frame(temp_path, task, speak_result=False)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Testing on Android Device

1. Enable USB Debugging on Android device
2. Connect via USB
3. Install APK:
   ```bash
   adb install bin/visualassistant-1.0.0-debug.apk
   ```

## Important Considerations for Mobile

### Performance Optimization

- Reduce image resolution before processing
- Implement request throttling (don't process every frame)
- Use background threads for API calls
- Cache results when appropriate

### Battery Life

- Limit continuous camera usage
- Process frames at intervals (every 2-3 seconds)
- Allow users to pause/resume

### Privacy & Security

- Don't store images permanently
- Use HTTPS for all API calls
- Request only necessary permissions
- Provide clear privacy policy

### Offline Capability

Consider implementing:
- Local caching of common responses
- Queue requests when offline
- Basic offline object detection with TensorFlow Lite

## Troubleshooting

### Common Issues

1. **"Build failed - NDK not found"**
   ```bash
   export ANDROID_NDK=/path/to/ndk
   ```

2. **"Camera permission denied"**
   - Ensure permissions are in buildozer.spec
   - Grant permissions on first app launch

3. **"API connection failed on mobile"**
   - Check mobile data/WiFi connection
   - Verify API keys work from mobile network

## Publishing to Play Store

1. Create Google Play Developer account ($25 one-time)
2. Generate signed release APK
3. Create store listing with screenshots
4. Submit for review

### Requirements for Accessibility Apps

- Clearly describe accessibility features
- Include demo video showing usage
- Provide contact for accessibility feedback
- Follow Android accessibility guidelines

---

**Note**: Building Android APKs requires significant setup. For quick testing, consider using the Windows version or running the Python code directly on a Linux/Mac system.
