# Visual Assistant for Visually Impaired

A powerful AI-powered visual assistance system designed to help visually impaired people with object detection, navigation, text reading (OCR), and face recognition.

## Features

### 🎯 Core Capabilities

1. **Object Detection** - Identify and describe objects in the environment with their positions
2. **Navigation Assist** - Real-time navigation guidance with obstacle detection
3. **Text Reading (OCR)** - Extract and read text from signs, documents, labels using NVIDIA Nemotron OCR
4. **Face Recognition** - Describe people, their positions, and distinctive features

### 🤖 AI Models Used

- **Vision Analysis**: Meta Llama 3.2 90B Vision Instruct
- **OCR**: NVIDIA Nemotron OCR v1
- **Text Processing**: MiniMax M2.7 (for cleaning and normalizing OCR text)
- **Text-to-Speech**: NVIDIA Magpie TTS Multilingual

### 🎙️ Voice Features

- Multiple voice profiles (Default, Calm Female, Clear Male, Expressive)
- Automatic expression adjustment based on content
- Context-aware voice selection:
  - Emergency/warning messages → Expressive voice with higher pitch
  - Navigation instructions → Clear male voice
  - Text reading → Calm female voice
  - General descriptions → Default balanced voice

## Installation

### Requirements

- Python 3.8 or higher
- Windows 10/11 (for GUI)
- Webcam (optional, for live capture)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application (Recommended)

```bash
python gui.py
```

The GUI provides:
- Easy task selection with large, accessible buttons
- Image loading from file or camera capture
- Voice settings customization
- Real-time results display
- Audio playback of analysis

### Command Line Usage

```bash
python core.py
```

Or use programmatically:

```python
from core import VisualAssistant

assistant = VisualAssistant()

# Process an image for object detection
result = assistant.process_frame("path/to/image.jpg", task_type="detection")
print(result["analysis"])

# Read text from an image
result = assistant.process_frame("path/to/text.jpg", task_type="ocr")
print(result["analysis"])

# Get navigation assistance
result = assistant.process_frame("path/to/scene.jpg", task_type="navigation")
print(result["analysis"])
```

### Keyboard Shortcuts (GUI)

- `Alt+L` - Load image
- `Alt+A` or `F5` - Analyze current image
- `Tab` - Navigate between controls

## Project Structure

```
visual_assistant/
├── core.py              # Core AI functionality
├── gui.py               # Graphical user interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── run.bat             # Windows launcher script
```

## API Keys

This application uses NVIDIA NIM APIs. The API keys are included in the code:
- Llama 3.2 Vision API
- NVIDIA Nemotron OCR API
- MiniMax LLM API
- NVIDIA Magpie TTS API

**Note**: For production use, you should obtain your own API keys from [NVIDIA Build](https://build.nvidia.com/).

## Accessibility Features

- High contrast UI design
- Large, clear fonts
- Keyboard navigation support
- Screen reader friendly
- Audio feedback for all actions
- Voice customization options

## Live Camera Mode

The application can capture images from your webcam for real-time assistance:

1. Click "Use Camera" button
2. The app will capture a photo
3. Select your desired task
4. Click "Analyze & Speak"

## Production Deployment

### Creating a Standalone Executable (Windows)

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile --windowed --name="VisualAssistant" gui.py
```

The executable will be created in the `dist` folder.

### Building Android APK

For Android deployment, you'll need to:
1. Use Kivy or BeeWare for cross-platform mobile development
2. Adapt the core logic for mobile cameras
3. Package using buildozer or similar tools

See `android_build_guide.md` for detailed instructions.

## Troubleshooting

### Common Issues

1. **"Cannot access camera"**
   - Ensure camera permissions are granted
   - Check if another application is using the camera
   - Try reconnecting the webcam

2. **"API Error" or "Connection Failed"**
   - Check your internet connection
   - Verify API keys are valid
   - Check NVIDIA API service status

3. **"Audio playback failed"**
   - Ensure speakers/headphones are connected
   - Check system volume settings
   - Verify pygame installation

## Safety Notice

⚠️ **Important**: This tool is designed to assist visually impaired individuals but should not be relied upon as the sole means of navigation or safety. Always:
- Use traditional mobility aids (cane, guide dog)
- Exercise caution in unfamiliar environments
- Don't rely solely on AI for critical safety decisions

## License

This project is provided as-is for educational and assistive purposes.

## Support

For issues, questions, or contributions, please open an issue on the project repository.

---

**Made with ❤️ for accessibility and inclusion**
