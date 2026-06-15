"""
Eyeva AI – Voice Activity Detection Service
Wraps Silero VAD for real-time speech start and end detection.
"""
import time
from typing import Optional
import torch
import numpy as np
from loguru import logger
from silero_vad import load_silero_vad, VADIterator

_vad_model = None

def get_model():
    """Lazily load the Silero VAD model."""
    global _vad_model
    if _vad_model is None:
        logger.info("Loading Silero VAD model...")
        _vad_model = load_silero_vad()
        logger.info("Silero VAD model loaded successfully")
    return _vad_model


def init_model():
    """Eagerly load the Silero VAD model on startup."""
    get_model()


class SpeechDetector:
    """Class to track speech state and events for a streaming PCM audio session."""

    def __init__(self, threshold: float = 0.5, min_silence_duration_ms: int = 400):
        model = get_model()
        # VADIterator expects a float32 tensor chunk of size 512, 1024, or 1536 at 16000Hz.
        self.iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=16000,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=30
        )
        self.is_speech = False

    def process_chunk(self, chunk: np.ndarray) -> tuple[Optional[bool], float]:
        """
        Process a chunk of 16kHz float32 raw PCM.
        
        Returns:
            (speech_event, latency_ms)
            speech_event:
                True: Speech start detected
                False: Speech end detected
                None: No state change
        """
        start_time = time.monotonic()
        
        # Convert numpy array to torch float32 tensor
        if not isinstance(chunk, torch.Tensor):
            tensor_chunk = torch.from_numpy(chunk).float()
        else:
            tensor_chunk = chunk.float()
            
        speech_dict = self.iterator(tensor_chunk)
        latency_ms = (time.monotonic() - start_time) * 1000
        
        if speech_dict:
            if 'start' in speech_dict:
                logger.info("VAD: Speech start detected")
                self.is_speech = True
                return True, latency_ms
            elif 'end' in speech_dict:
                logger.info("VAD: Speech end detected")
                self.is_speech = False
                return False, latency_ms
                
        return None, latency_ms

    def reset(self):
        """Reset internal iterator states between recordings."""
        self.iterator.reset_states()
        self.is_speech = False
