# Quick Start Guide

## For Windows Users (Easiest Method)

### Step 1: Install Python
1. Download Python from https://python.org/downloads/
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Click "Install Now"

### Step 2: Setup Visual Assistant
1. Extract the `visual_assistant` folder
2. Double-click `setup.bat`
3. Wait for installation to complete
4. When prompted, choose "Y" to launch the app

### Step 3: Use the Application
1. Click "Load Image" to select an image file
   OR
   Click "Use Camera" to capture from webcam
2. Select your task:
   - 🔍 Object Detection - Identify objects in the scene
   - 🧭 Navigation Assist - Get directions and obstacle warnings
   - 📄 Read Text (OCR) - Read text from signs, documents, labels
   - 👤 Face Recognition - Describe people in the image
3. Click "Analyze & Speak"
4. Listen to the audio description!

## Keyboard Shortcuts
- `Alt+L` - Load Image
- `Alt+A` or `F5` - Analyze current image
- `Tab` - Navigate between buttons

## Voice Settings
- Choose from 4 voice profiles
- Auto-expressions automatically adjust tone based on content:
  - Warnings → More urgent voice
  - Navigation → Clear, direct voice
  - Reading → Calm, soothing voice

---

## For Advanced Users

### Command Line Usage
```bash
cd visual_assistant
python gui.py
```

### Test Installation
```bash
python test_system.py
```

### Build Standalone Executable
```bash
python build_exe.py
```

The executable will be created in the `dist` folder.

---

## Troubleshooting

**Problem**: "Python not found"
- Solution: Reinstall Python and make sure to check "Add to PATH"

**Problem**: "Cannot access camera"
- Solution: Grant camera permissions, close other apps using camera

**Problem**: "API Error"
- Solution: Check internet connection, verify API keys are valid

**Problem**: "No audio"
- Solution: Check system volume, ensure speakers/headphones connected

---

For detailed documentation, see README.md
For Android APK building, see android_build_guide.md
