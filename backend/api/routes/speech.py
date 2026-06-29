"""
Eyeva AI – Speech Routes

POST /api/speech/transcribe  – audio file → text (multipart)
POST /api/speech/synthesize  – text → WAV audio (binary response)
WS   /ws/speech              – full-duplex voice assistant loop:
                               audio in → transcript → VLM response → TTS audio out
"""
import asyncio
import json
import time
import re
import base64
import io
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger

from models.schemas import SynthesizeRequest, TranscribeResponse, WSMessage
from services import stt_service, tts_service, vision_service, vad_service, ocr_service
from websocket.manager import manager

router = APIRouter(tags=["speech"])


# ── REST: Transcribe ──────────────────────────────────────────────────────────

@router.post(
    "/api/speech/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe audio to text",
)
async def transcribe_audio(audio: UploadFile = File(...)) -> TranscribeResponse:
    """
    Transcribe an uploaded audio file (WAV, MP3, WebM, OGG) to text using Faster Whisper.
    Language is auto-detected.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file.")

    try:
        transcript, lang, confidence, latency = await stt_service.transcribe_audio(audio_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TranscribeResponse(
        transcript=transcript,
        language=lang,
        confidence=confidence,
        latency_ms=latency,
    )


# ── REST: Synthesize ──────────────────────────────────────────────────────────

@router.post(
    "/api/speech/synthesize",
    summary="Synthesize text to speech",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize_speech(req: SynthesizeRequest) -> Response:
    """
    Convert text to WAV audio using Kokoro TTS.
    Returns raw WAV bytes with Content-Type: audio/wav.
    """
    try:
        wav_bytes, latency = await tts_service.synthesize_speech(
            req.text, req.voice, req.speed
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Latency-Ms": str(round(latency, 1))},
    )


# ── Background Worker, Helper Functions and VAD State Machine ──────────────────

def compute_image_difference(img_b64_1: str, img_b64_2: str) -> float:
    """
    Calculate the Mean Squared Error (MSE) difference between two base64-encoded images.
    Converts images to grayscale and resizes them to 64x64 for fast and robust comparison.
    """
    try:
        def to_array(b64_str):
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
            img = img.resize((64, 64))
            return np.array(img, dtype=np.float32)

        arr1 = to_array(img_b64_1)
        arr2 = to_array(img_b64_2)
        mse = float(np.mean((arr1 - arr2) ** 2))
        return mse
    except Exception as e:
        logger.warning("Error computing image difference: {}", e)
        return float('inf')


def extract_objects_and_people(description: str) -> tuple[list[str], int]:
    """
    Extract a list of candidate objects and count of people from the VLM scene description.
    Uses basic regex/NLP heuristic.
    """
    desc_lower = description.lower()
    
    # 1. Count people
    plural_people_words = ["people", "men", "women", "guys", "girls", "boys", "children", "individuals", "babies", "toddlers"]
    
    num_map = {
        "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "several": 3,
        "few": 2, "many": 5, "some": 3
    }
    
    people_count = 0
    pattern = r'\b(\d+|one|a|an|two|three|four|five|six|seven|eight|nine|ten|several|few|many|some)?\s*(people|person|man|woman|men|women|child|children|guy|guys|girl|girls|boy|boys|individual|individuals|baby|babies|toddler|toddlers)\b'
    matches = re.findall(pattern, desc_lower)
    
    for num_str, noun in matches:
        count = 1
        if num_str:
            num_str = num_str.strip()
            if num_str.isdigit():
                count = int(num_str)
            elif num_str in num_map:
                count = num_map[num_str]
        else:
            if noun in plural_people_words:
                count = 2
        people_count += count
    
    # 2. Extract objects (non-stopword nouns/adjectives)
    stopwords = {
        "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
        "doing", "and", "but", "or", "because", "as", "until", "while", "of", "at",
        "by", "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "eyeva", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        "scene", "image", "picture", "photo", "background", "foreground", "left", "right",
        "ahead", "front", "behind", "middle", "center", "shows", "contains", "there is", "there are"
    }
    
    clean_desc = re.sub(r'[^\w\s]', ' ', desc_lower)
    words = clean_desc.split()
    candidate_objects = []
    for w in words:
        if w not in stopwords and len(w) > 2 and not w.isdigit():
            if w not in candidate_objects:
                candidate_objects.append(w)
                
    return candidate_objects, people_count


def check_scene_memory_for_query(query: str, scene_memory: dict) -> Optional[str]:
    """
    Check if the user query can be answered using the cached scene memory.
    Returns the answer string if hit, or None if a fresh VLM call is required.
    """
    if not scene_memory or not scene_memory.get("scene_summary"):
        return None
        
    query_lower = query.lower()
    last_update = scene_memory.get("last_update", 0.0)
    age = time.monotonic() - last_update
    is_fresh = age < 3.0
    
    # 1. OCR / Read queries
    ocr_keywords = ["read", "text", "writing", "sign", "book", "label", "word", "written"]
    if any(k in query_lower for k in ocr_keywords):
        detected_text = scene_memory.get("detected_text", "").strip()
        if detected_text:
            return f"The text detected is: {detected_text}"
        elif is_fresh:
            return "I don't see any text in the scene."
        return None

    # 2. People queries
    people_keywords = ["person", "people", "anyone", "anybody", "someone", "somebody", "man", "woman", "men", "women", "guy", "girl", "boy", "child"]
    if any(k in query_lower for k in people_keywords):
        count = scene_memory.get("people_count", 0)
        if count > 0:
            p_word = "person" if count == 1 else "people"
            return f"Yes, I see {count} {p_word} in front of you."
        elif is_fresh:
            return "No, I don't see anyone in the scene right now."
        return None

    # 3. Specific object query
    cached_objects = scene_memory.get("objects", [])
    for obj in cached_objects:
        if re.search(r'\b' + re.escape(obj) + r'\b', query_lower):
            return f"Yes, I see a {obj} in the scene."
            
    common_objects = ["laptop", "phone", "bottle", "cup", "mug", "glass", "chair", "table", "keyboard", "mouse", "screen", "monitor", "pen", "notebook", "backpack", "bag", "door", "window", "plant", "key", "wallet"]
    for obj in common_objects:
        if re.search(r'\b' + re.escape(obj) + r'\b', query_lower):
            if is_fresh:
                return f"No, I don't see a {obj} in the scene."
            return None

    # 4. General scene queries (if fresh)
    # Match ONLY general keywords specifically, avoiding catching detail queries like "is the fan moving"
    general_keywords = ["what is in front", "what's in front", "what do you see", "describe the scene", "what's there", "what is there", "what is around", "where am i", "look at", "overview", "describe"]
    if any(k in query_lower for k in general_keywords):
        if is_fresh:
            return scene_memory.get("scene_summary")
            
    return None


def is_conversational_query(text: str) -> bool:
    """Check if a query is purely conversational/text-only and doesn't need vision analysis."""
    text_lower = text.lower()
    
    # Heuristics for visual queries (words pointing to immediate vision/OCR needs)
    visual_keywords = [
        "see", "look", "describe", "image", "photo", "camera", "picture", 
        "color", "what is this", "what are these", "read", "writing", "text", 
        "sign", "label", "book", "words", "money", "bill", "note", "dollar",
        "obstacle", "hazard", "ahead", "front", "navigation", "where to", "path"
    ]
    
    # If any visual keywords are in the query, it requires vision.
    if any(k in text_lower for k in visual_keywords):
        return False
        
    # Heuristics for common conversational queries
    conversational_patterns = [
        "hello", "hi", "how are you", "who are you", "what is your name",
        "thank you", "thanks", "tell me a joke", "tell a story", "what is the capital",
        "weather", "time", "date", "math", "calculate", "capital of", "who wrote",
        "explain", "define", "what does mean", "help me with"
    ]
    if any(k in text_lower for k in conversational_patterns):
        return True
        
    # Default: if it is very short and generic conversational phrase
    words = text_lower.split()
    if len(words) <= 3 and any(w in {"hey", "eyeva", "assistant", "ok", "okay", "yes", "no"} for w in words):
        return True
        
    # By default, assume it might need vision if not classified as conversational
    return False



async def vision_worker_loop(websocket: WebSocket, session) -> None:
    """
    Background task that continuously analyzes incoming camera frames,
    detects motion/stability, updates scene memory, and runs OCR.
    """
    logger.info("Starting background vision worker loop for session {}", session.session_id)
    await asyncio.sleep(0.5)
    
    while True:
        try:
            # Pause background vision worker when user is actively interacting to prioritize quota and avoid rate limits
            if session.status != "idle":
                await asyncio.sleep(1.0)
                continue

            frame_b64 = session.last_frame_b64
            if not frame_b64:
                await asyncio.sleep(0.5)
                continue
                
            last_analyzed = session.last_analyzed_frame_b64
            analyze_needed = True
            interval = 2.5
            
            if last_analyzed:
                diff = compute_image_difference(frame_b64, last_analyzed)
                logger.debug("Background vision: frame difference (MSE) = {:.2f}", diff)
                
                # Rate limit protection: skip VLM/OCR calls if scene is stable (< 15 MSE) and memory is cached
                if diff < 15.0 and session.scene_memory.get("scene_summary"):
                    analyze_needed = False
                    interval = 2.5
                    logger.debug("Background vision: negligible change and memory cached, skipping analysis.")
                elif diff >= 15.0:
                    interval = 1.0
                    logger.debug("Background vision: motion detected, setting interval to 1s")
                else:
                    interval = 2.5
                    logger.debug("Background vision: stable scene, setting interval to 2.5s")
            
            if analyze_needed:
                start_time = time.monotonic()
                try:
                    # Prompt VLM to do both scene description and text transcription in a single call to save 50% API calls
                    vlm_prompt = (
                        "Describe what you see in the image clearly and concisely for a visually impaired user in 1-2 short sentences. "
                        "Do not use markdown. Do not say 'I see' or 'The image shows'. "
                        "Also, transcribe any text visible in the image exactly as it appears. If no text is found, write 'None'.\n\n"
                        "Format your response exactly as:\n"
                        "Description: <scene description>\n"
                        "Text: <transcribed text>"
                    )
                    
                    description, _, _ = await vision_service.analyze_frame(
                        frame_b64,
                        prompt=vlm_prompt,
                        use_cache=False
                    )
                    
                    # Parse combined VLM response
                    scene_desc = ""
                    ocr_text = ""
                    if "Description:" in description and "Text:" in description:
                        parts = description.split("Text:", 1)
                        scene_desc = parts[0].replace("Description:", "").strip()
                        text_part = parts[1].strip()
                        ocr_text = text_part if text_part.lower() != "none" else ""
                    else:
                        scene_desc = description
                        ocr_text = ""
                        
                    if scene_desc:
                        obj_list, people_cnt = extract_objects_and_people(scene_desc)
                        session.scene_memory.update({
                            "timestamp": time.time(),
                            "last_update": time.monotonic(),
                            "scene_summary": scene_desc,
                            "summary": scene_desc,
                            "objects": obj_list,
                            "people_count": people_cnt,
                            "detected_text": ocr_text,
                            "text": ocr_text
                        })
                        session.last_analyzed_frame_b64 = frame_b64
                        
                        if ocr_text.strip():
                            await manager.send_ocr_result(websocket, ocr_text)
                            
                        logger.info("Scene memory updated in {:.1f}ms (combined VLM call): {} objects, {} people, OCR text: '{}'", 
                                    (time.monotonic() - start_time) * 1000, len(obj_list), people_cnt, ocr_text[:30])
                                    
                except Exception as e:
                    logger.error("Error in background VLM/OCR execution: {}", e)
                    
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info("Background vision worker loop cancelled for session {}", session.session_id)
            break
        except Exception as e:
            logger.error("Unexpected error in background vision worker: {}", e)
            await asyncio.sleep(2.5)


# ── WebSocket: Full-duplex voice assistant ────────────────────────────────────

@router.websocket("/ws/speech")
async def speech_websocket(websocket: WebSocket) -> None:
    """
    Full-duplex voice assistant WebSocket supporting low-latency raw Float32 PCM streaming,
    backend Silero Voice Activity Detection (VAD) auto-triggering, real-time background transcription updates,
    and concurrent sentence-level VLM -> Kokoro TTS streaming response generation.
    """
    session = await manager.connect(websocket)
    await manager.send_status(websocket, "idle")

    # Raw audio float32 sample buffer
    audio_buffer: list[float] = []
    active_task: Optional[asyncio.Task] = None
    transcribe_task: Optional[asyncio.Task] = None
    vision_task: Optional[asyncio.Task] = None
    
    # State tracking
    is_listening_mode = False
    speech_start_time = 0.0
    vad_detector = vad_service.SpeechDetector()
    vad_buffer = np.array([], dtype=np.float32)
    VAD_CHUNK_SIZE = 512

    # Background vision worker is disabled to save API key rate limits and run direct VLM query on start
    vision_task = None

    async def transcribe_stream_loop():
        nonlocal audio_buffer, is_listening_mode
        last_transcribed_len = 0
        while True:
            await asyncio.sleep(0.8)
            if not is_listening_mode:
                break
                
            current_audio = np.array(audio_buffer, dtype=np.float32)
            if len(current_audio) - last_transcribed_len > 8000:
                last_transcribed_len = len(current_audio)
                try:
                    transcript, _, _, _ = await stt_service.transcribe_audio_numpy(
                        current_audio, language=session.lang
                    )
                    if transcript.strip() and is_listening_mode:
                        await manager.send_transcript(websocket, transcript)
                except Exception as e:
                    logger.debug("Streaming transcription background error: {}", e)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                manager.disconnect(websocket)
                break

            # ── Binary: Raw PCM float32 samples streaming ──────────────────────
            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                samples = np.frombuffer(chunk_bytes, dtype=np.float32)

                # Feed samples to Silero VAD
                vad_buffer = np.concatenate([vad_buffer, samples])
                while len(vad_buffer) >= VAD_CHUNK_SIZE:
                    vad_chunk = vad_buffer[:VAD_CHUNK_SIZE]
                    vad_buffer = vad_buffer[VAD_CHUNK_SIZE:]

                    event, vad_lat = vad_detector.process_chunk(vad_chunk)
                    
                    if event is True:
                        logger.info("VAD: Speech start detected in {:.1f}ms", vad_lat)
                        
                        # Interrupt active VLM/TTS processing if new speech begins
                        if active_task and not active_task.done():
                            logger.info("Interrupting active response generation for session {} due to incoming speech.", session.session_id)
                            active_task.cancel()
                            active_task = None
                        
                        session.status = "listening"
                        await manager.send_status(websocket, "listening")
                        is_listening_mode = True
                        audio_buffer = list(vad_chunk)
                        speech_start_time = time.monotonic()
                        
                        # Launch background real-time speech transcription loop
                        if transcribe_task and not transcribe_task.done():
                            transcribe_task.cancel()
                        transcribe_task = asyncio.create_task(transcribe_stream_loop())
                        
                    elif event is False:
                        logger.info("VAD: Speech end detected in {:.1f}ms", vad_lat)
                        if session.status == "listening":
                            session.status = "processing"
                            await manager.send_status(websocket, "processing")
                            is_listening_mode = False
                            
                            if transcribe_task and not transcribe_task.done():
                                transcribe_task.cancel()
                                
                            speech_end_time = time.monotonic()
                            
                            # Immediately trigger downstream pipeline (Whisper STT -> VLM -> TTS)
                            active_task = asyncio.create_task(
                                _process_audio_numpy(
                                    websocket,
                                    session,
                                    np.array(audio_buffer, dtype=np.float32),
                                    speech_end_time,
                                    vad_lat
                                )
                            )
                            audio_buffer = []
                    else:
                        # Accumulate samples if we are currently listening
                        if session.status == "listening":
                            audio_buffer.extend(vad_chunk.tolist())
                continue

            # ── Text: Control JSON messages ───────────────────────────────────
            if "text" in message and message["text"]:
                try:
                    msg = WSMessage(**json.loads(message["text"]))
                except Exception:
                    await manager.send_error(websocket, "Invalid message format.")
                    continue

                if msg.type == "config":
                    if msg.voice:
                        session.voice = msg.voice
                    if msg.speed:
                        session.speed = msg.speed
                    if msg.lang:
                        session.lang = msg.lang
                    if msg.mode:
                        session.mode = msg.mode
                    if msg.vad_threshold is not None:
                        session.vad_threshold = msg.vad_threshold
                        logger.info("Re-initializing VAD detector with threshold={}", session.vad_threshold)
                        vad_detector = vad_service.SpeechDetector(threshold=session.vad_threshold)

                elif msg.type == "frame":
                    session.last_frame_b64 = msg.data if msg.data else None

                elif msg.type == "command":
                    action = msg.action or ""

                    if action == "transcribe":
                        is_listening_mode = False
                        if transcribe_task and not transcribe_task.done():
                            transcribe_task.cancel()

                        if active_task and not active_task.done():
                            logger.info("Interrupting active task for session {} due to transcribe command.", session.session_id)
                            active_task.cancel()
                            active_task = None

                        if audio_buffer:
                            speech_end_time = time.monotonic()
                            session.status = "processing"
                            await manager.send_status(websocket, "processing")
                            active_task = asyncio.create_task(
                                _process_audio_numpy(
                                    websocket,
                                    session,
                                    np.array(audio_buffer, dtype=np.float32),
                                    speech_end_time,
                                    0.0
                                )
                            )
                            audio_buffer = []
                        else:
                            session.status = "idle"
                            await manager.send_status(websocket, "idle")

                    elif action == "stop":
                        is_listening_mode = False
                        if transcribe_task and not transcribe_task.done():
                            transcribe_task.cancel()

                        if active_task and not active_task.done():
                            logger.info("Interrupting active task for session {} due to stop command.", session.session_id)
                            active_task.cancel()
                            active_task = None
                        audio_buffer = []
                        session.status = "idle"
                        await manager.send_status(websocket, "idle")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("Speech WS unexpected error: {}", e)
        try:
            await manager.send_error(websocket, "An unexpected error occurred.")
        except Exception:
            pass
        manager.disconnect(websocket)
    finally:
        is_listening_mode = False
        session.status = "idle"
        if vision_task and not vision_task.done():
            vision_task.cancel()
        if transcribe_task and not transcribe_task.done():
            transcribe_task.cancel()
        if active_task and not active_task.done():
            active_task.cancel()


async def _process_audio_numpy(
    websocket: WebSocket,
    session,
    audio_numpy: np.ndarray,
    speech_end_time: float,
    vad_lat: float
) -> None:
    """
    Downstream processing pipeline: transcribes raw float32 mono PCM, streams AI answers,
    synthesizes speech concurrently sentence-by-sentence, and streams WAV bytes back.
    Logs latency performance metrics.
    """
    try:
        metrics = {
            "vad_detection_time": vad_lat,
            "transcription_complete": None,
            "ai_request_start": None,
            "ai_first_token": None,
            "tts_start": None,
            "first_audio_sent": None,
            "response_complete": None
        }

        session.status = "processing"
        await manager.send_status(websocket, "processing")

        # 1. Speech-to-Text (Whisper)
        try:
            transcript, lang, _, stt_lat = await stt_service.transcribe_audio_numpy(
                audio_numpy, language=session.lang
            )
            metrics["transcription_complete"] = time.monotonic()
        except RuntimeError as e:
            await manager.send_error(websocket, str(e))
            session.status = "idle"
            await manager.send_status(websocket, "idle")
            return

        if not transcript.strip():
            logger.info("VAD/STT completed: empty transcription, returning to idle state.")
            session.status = "idle"
            await manager.send_status(websocket, "idle")
            return

        # Check if the transcript contains the wake word "eva" or "eyeva"
        # Strong wake words: can appear anywhere
        # Weak wake words: must appear at the start of the transcript
        transcript_lower = transcript.lower()
        has_strong_ww = bool(re.search(r'\b(eva|eyeva|ava|iva|ifa|ayeva)\b', transcript_lower))
        has_weak_ww = bool(re.search(r'^\s*(?:hey\s+|ok\s+|okay\s+)?(?:even|ever|every)\b', transcript_lower))

        if not (has_strong_ww or has_weak_ww):
            logger.info("Ignoring query because wake word was not detected: '{}'", transcript)
            session.status = "idle"
            await manager.send_status(websocket, "idle")
            return

        # Clean/strip the wake word and preceding/succeeding filler/punctuation
        cleaned_transcript = re.sub(
            r'\b(?:hey\s+|ok\s+|okay\s+)?(?:eva|eyeva|ava|iva|ifa|ayeva)\b',
            '',
            transcript,
            flags=re.IGNORECASE
        )
        cleaned_transcript = re.sub(
            r'^\s*(?:hey\s+|ok\s+|okay\s+)?(?:even|ever|every)\b',
            '',
            cleaned_transcript,
            flags=re.IGNORECASE
        )
        cleaned_transcript = cleaned_transcript.strip(".,!?:;— ")
        cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript)

        logger.info("Original transcript: '{}' | Cleaned transcript: '{}'", transcript, cleaned_transcript)

        # If they just said the wake word, respond with "Yes, I am listening."
        if not cleaned_transcript:
            logger.info("Wake word only detected. Responding directly.")
            response_text = "Yes, I am listening."
            session.status = "speaking"
            await manager.send_status(websocket, "speaking")
            await manager.send_response(websocket, response_text)
            
            try:
                async for chunk_bytes in tts_service.stream_synthesize_speech(
                    response_text, session.voice, session.speed
                ):
                    await manager.send_bytes(websocket, chunk_bytes)
            except Exception as e:
                logger.error("Speech synthesis failed for wake word response: {}", e)
            
            session.status = "idle"
            await manager.send_status(websocket, "idle")
            return

        await manager.send_transcript(websocket, cleaned_transcript)
        logger.info("Transcript: '{}'", cleaned_transcript)

        # 2. Generative response & streaming TTS pipeline
        session.status = "speaking"
        await manager.send_status(websocket, "speaking")
        metrics["ai_request_start"] = time.monotonic()

        logger.info("Direct VLM query='{}' (mode={})", cleaned_transcript, session.mode)
        try:
            # Reuse latest frame buffer if present (multimodal VLM handles both vision and general queries)
            if session.last_frame_b64:
                if session.mode == "navigation":
                    nav_prompt = (
                        f"You are Eyeva, a navigation and obstacle-avoidance assistant for visually impaired users. "
                        f"Look at the image and answer this query: '{cleaned_transcript}'. "
                        f"Prioritize identifying any immediate obstacles, safe paths, or walking directions. "
                        f"Keep it very brief — limit your response to 1-2 short sentences maximum."
                    )
                    token_stream = vision_service.stream_answer_question(
                        session.last_frame_b64, nav_prompt
                    )
                elif session.mode == "money":
                    money_prompt = (
                        f"You are Eyeva, a money recognition assistant. "
                        f"Look at the image and answer this query: '{cleaned_transcript}'. "
                        f"Identify and count any banknotes or coins. State the total value clearly. "
                        f"Keep it extremely concise — limit your response to 1 short sentence."
                    )
                    token_stream = vision_service.stream_answer_question(
                        session.last_frame_b64, money_prompt
                    )
                else:
                    # Default voice/scene VLM query
                    token_stream = vision_service.stream_answer_question(
                        session.last_frame_b64, cleaned_transcript
                    )
            else:
                logger.info("Routing query='{}' to text-only LLM (no image frame available)", cleaned_transcript)
                token_stream = vision_service.stream_answer_text_query(cleaned_transcript)
        except Exception as e:
            logger.error("Response generation stream setup failed: {}", e)
            async def fallback_stream():
                yield "I'm sorry, I had trouble answering that. Could you repeat?"
            token_stream = fallback_stream()

        # Helper async generator that yields complete sentences from the token stream
        async def sentence_generator_from_stream():
            buffer = ""
            sentence_endings = {".", "!", "?", "\n"}
            full_response_text = []

            try:
                async for token in token_stream:
                    if metrics["ai_first_token"] is None:
                        metrics["ai_first_token"] = time.monotonic()

                    buffer += token
                    full_response_text.append(token)

                    while True:
                        split_idx = -1
                        # Check sentence endings
                        for i, char in enumerate(buffer):
                            if char in sentence_endings:
                                if i + 1 < len(buffer) and buffer[i+1].isspace():
                                    split_idx = i + 1
                                    break
                                elif i + 1 == len(buffer):
                                    split_idx = i + 1
                                    break

                        if split_idx != -1:
                            sentence = buffer[:split_idx].strip()
                            buffer = buffer[split_idx:]
                            if sentence:
                                yield sentence
                        else:
                            break
            except Exception as e:
                logger.error("Error during token stream iteration: {}", e)
                if not full_response_text:
                    yield "I'm sorry, I encountered an API rate limit or connection error. Please try again in a moment."
                else:
                    yield "I had trouble finishing my response."
                return

            # Yield remaining buffer text
            remaining = buffer.strip()
            if remaining:
                yield remaining

            # Send the complete text response back to the client at the end
            full_text = "".join(full_response_text).strip()
            await manager.send_response(websocket, full_text)

        generator = sentence_generator_from_stream()

        # Concurrent TTS pipeline queue to pre-synthesize downstream sentences in background tasks
        task_queue = asyncio.Queue()

        async def tts_producer():
            try:
                async for sentence in generator:
                    text = sentence.strip()
                    if not text:
                        continue

                    # Spawn background task to synthesize immediately
                    async def run_synthesis(txt):
                        try:
                            wav_bytes, _ = await tts_service.synthesize_speech(
                                txt, session.voice, session.speed
                            )
                            return wav_bytes
                        except Exception as e:
                            logger.error("Background TTS synthesis failed for '{}': {}", txt, e)
                            return None

                    task = asyncio.create_task(run_synthesis(text))
                    await task_queue.put(task)
            finally:
                # Put None to signal end of stream
                await task_queue.put(None)

        producer_task = asyncio.create_task(tts_producer())

        try:
            while True:
                task = await task_queue.get()
                if task is None:
                    break

                if metrics["tts_start"] is None:
                    metrics["tts_start"] = time.monotonic()

                wav_bytes = await task
                if wav_bytes:
                    if metrics["first_audio_sent"] is None:
                        metrics["first_audio_sent"] = time.monotonic()
                    await manager.send_bytes(websocket, wav_bytes)
        finally:
            if not producer_task.done():
                producer_task.cancel()

        metrics["response_complete"] = time.monotonic()

        # Print latency summary metrics in the backend logs
        stt_ms = (metrics["transcription_complete"] - speech_end_time) * 1000
        ai_conn_ms = (metrics["ai_request_start"] - metrics["transcription_complete"]) * 1000
        ai_first_token_ms = ((metrics["ai_first_token"] - metrics["ai_request_start"]) * 1000) if metrics["ai_first_token"] else 0.0
        tts_start_ms = ((metrics["tts_start"] - metrics["ai_first_token"]) * 1000) if (metrics["tts_start"] and metrics["ai_first_token"]) else 0.0
        first_audio_ms = ((metrics["first_audio_sent"] - metrics["tts_start"]) * 1000) if (metrics["first_audio_sent"] and metrics["tts_start"]) else 0.0
        resp_complete_ms = ((metrics["response_complete"] - metrics["first_audio_sent"]) * 1000) if (metrics["response_complete"] and metrics["first_audio_sent"]) else 0.0
        total_perceived_ms = ((metrics["first_audio_sent"] - speech_end_time) * 1000) if metrics["first_audio_sent"] else 0.0

        logger.info(
            "\n====================================\n"
            "   AUDIO PIPELINE LATENCY METRICS   \n"
            "====================================\n"
            f"VAD Detection    : {metrics['vad_detection_time']:.1f}ms\n"
            f"STT (Whisper)    : {stt_ms:.1f}ms\n"
            f"AI Request Start : {ai_conn_ms:.1f}ms\n"
            f"AI First Token   : {ai_first_token_ms:.1f}ms\n"
            f"TTS Start        : {tts_start_ms:.1f}ms\n"
            f"First Audio Sent : {first_audio_ms:.1f}ms\n"
            f"Response Complete: {resp_complete_ms:.1f}ms\n"
            f"Total Perceived  : {total_perceived_ms:.1f}ms (Speech End -> First Audio)\n"
            "===================================="
        )

        session.status = "idle"
        await manager.send_status(websocket, "idle")

    except asyncio.CancelledError:
        logger.info("Speech processing task cancelled for session {}", session.session_id)
        session.status = "idle"
        raise
    except Exception as e:
        logger.error("Speech processing loop error: {}", e)
        session.status = "idle"
        await manager.send_status(websocket, "idle")

