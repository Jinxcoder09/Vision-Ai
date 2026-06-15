# 👁️ Eyeva AI V1

**AI-powered visual assistant for visually impaired users.**

Eyeva AI uses your camera, voice, and cutting-edge AI to describe the world around you — reading text, identifying objects, and answering questions — all through natural voice interaction.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Voice Assistant** | Say "Hey Eyeva, what's in front of me?" |
| 🔍 **Scene Understanding** | Continuous AI analysis of your camera feed |
| 📖 **Text Reading** | OCR-powered reading of any text in view |
| ❓ **Q&A Mode** | Ask specific questions about what you see |
| ♿ **Fully Accessible** | Screen reader compatible, keyboard navigable, high contrast |

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docker.com) + [Docker Compose](https://docs.docker.com/compose/)
- NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com/)

### 1. Clone and configure

```bash
git clone <repo-url>
cd Vision-Ai
cp .env.example .env
```

Edit `.env` and add your API key:

```env
NVIDIA_API_KEY=your_key_here
```

### 2. Run

```bash
docker-compose up
```

> **First run** will download AI models (~2–4 GB). Subsequent starts are fast.

### 3. Open

Navigate to **http://localhost:3000**

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA NIM API key |
| `QWEN_API_KEY` | *(optional)* | Separate Qwen key; falls back to `NVIDIA_API_KEY` |
| `VLM_MODEL` | `qwen/qwen2.5-vl-7b-instruct` | Vision model ID |
| `WHISPER_MODEL` | `base.en` | Faster Whisper size (`tiny.en`→`large-v3`) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KOKORO_VOICE` | `af_heart` | Default TTS voice |
| `KOKORO_SPEED` | `1.0` | Speech speed (0.5–2.0) |
| `PADDLE_LANG` | `en` | OCR language (`en` or `hi`) |
| `MAX_FRAME_RATE` | `2.0` | Max vision API calls/sec |
| `RESPONSE_CACHE_TTL` | `30` | Cache duration in seconds |

See [`.env.example`](.env.example) for the full list.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Next.js 15)                  │
│  Dashboard │ Camera │ Text Reader │ Settings             │
└────────────────────┬───────────────────────────────────-─┘
                     │ WebSocket / REST
┌────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ STT Service  │  │ Vision Svc   │  │  OCR Service  │  │
│  │ Faster       │  │ NVIDIA NIM   │  │  PaddleOCR    │  │
│  │ Whisper      │  │ Qwen2.5-VL   │  │  EN + HI      │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │ TTS Service  │  │  WS Manager  │                      │
│  │ Kokoro TTS   │  │  + Cache     │                      │
│  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### Voice Pipeline
```
Microphone → Faster Whisper → Intent Detection
→ Qwen2.5-VL (if camera active) → Kokoro TTS → Speaker
```

### Text Reading Pipeline
```
Camera → PaddleOCR → Text Cleanup → Kokoro TTS → Speaker
```

---

## 🗂️ Project Structure

```
Vision-Ai/
├── backend/
│   ├── api/routes/         # REST + WebSocket endpoints
│   ├── services/           # AI model wrappers
│   ├── websocket/          # Connection manager
│   ├── core/               # Config, logging, cache
│   ├── models/             # Pydantic schemas
│   ├── main.py             # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/app/            # Next.js 15 App Router pages
│   ├── src/components/     # UI components
│   ├── src/hooks/          # useCamera, useVoice, useWebSocket
│   └── src/lib/            # API client, utilities
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🎙️ Voice Commands

| Command | Action |
|---|---|
| "Hey Eyeva, what's in front of me?" | Scene description |
| "Hey Eyeva, describe this room." | Full scene analysis |
| "Hey Eyeva, can you see any people?" | People detection |
| "Hey Eyeva, read this." | OCR text reading |
| "Hey Eyeva, what color is the bag?" | Specific Q&A |
| "Hey Eyeva, is there a door nearby?" | Object detection |

---

## 🛠️ Development (Without Docker)

### Backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install paddlepaddle  # CPU version
pip install -r requirements.txt

cp ../.env.example .env
# Edit .env with your API key

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
echo "NEXT_PUBLIC_WS_URL=ws://localhost:8000" >> .env.local
npm run dev
```

---

## ♿ Accessibility

- All interactive elements have `aria-label` attributes
- Full keyboard navigation (Tab, Enter, Space)
- `aria-live` regions for real-time AI responses
- High-contrast dark theme (WCAG AA compliant)
- Large touch targets (minimum 48×48px)
- Audible status confirmations

---

## ⚡ Performance Targets

| Operation | Target | Mechanism |
|---|---|---|
| STT transcription | < 1 second | Faster Whisper + VAD |
| Vision analysis | < 3 seconds | Frame throttling + cache |
| OCR extraction | < 2 seconds | PaddleOCR async executor |
| TTS synthesis | < 2 seconds | Kokoro streaming |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
