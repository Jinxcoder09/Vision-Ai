"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { parseWSMessage } from "@/lib/utils";
import { createSpeechWebSocket } from "@/lib/api";
import { AudioQueuePlayer } from "@/lib/audioQueue";

type VoiceStatus =
  | "idle"
  | "connecting"
  | "listening"
  | "processing"
  | "speaking"
  | "error";

interface TranscriptEntry {
  role: "user" | "ai";
  text: string;
  timestamp: number;
}

interface UseVoiceOptions {
  voice?: string;
  speed?: number;
  lang?: string;
  mode?: string;
  onTranscript?: (text: string) => void;
  onResponse?: (text: string) => void;
  onError?: (msg: string) => void;
  onStatusChange?: (status: VoiceStatus) => void;
}

interface UseVoiceReturn {
  status: VoiceStatus;
  transcript: TranscriptEntry[];
  latestResponse: string;
  isConnected: boolean;
  startListening: () => void;
  stopListening: () => void;
  sendFrame: (frameB64: string) => void;
  updateConfig: (cfg: { voice?: string; speed?: number; lang?: string; mode?: string; vad_threshold?: number }) => void;
  clearTranscript: () => void;
}

export function useVoice(options: UseVoiceOptions = {}): UseVoiceReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldReconnectRef = useRef(true);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioPlayerRef = useRef<AudioQueuePlayer | null>(null);

  // Initialize the player once on client side, clean up on unmount
  useEffect(() => {
    audioPlayerRef.current = new AudioQueuePlayer();
    return () => {
      audioPlayerRef.current?.stop();
    };
  }, []);
  const isRecordingRequestedRef = useRef(false);
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const statusRef = useRef(status);
  statusRef.current = status;
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [latestResponse, setLatestResponse] = useState("");
  const [isConnected, setIsConnected] = useState(false);

  // Store options in a ref to avoid recreation of hooks
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const updateStatus = useCallback(
    (s: VoiceStatus) => {
      setStatus(s);
      optionsRef.current.onStatusChange?.(s);
    },
    []
  );

  // ── Connect WebSocket ────────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("useVoice: connect called but WebSocket is already OPEN");
      return;
    }

    console.log("useVoice: Initiating WebSocket connection...");
    updateStatus("connecting");
    const ws = createSpeechWebSocket();
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("useVoice: WebSocket connection established");
      setIsConnected(true);
      updateStatus("idle");

      // Load settings from localStorage
      let localVoice = optionsRef.current.voice ?? "af_heart";
      let localSpeed = optionsRef.current.speed ?? 1.0;
      let localLang = optionsRef.current.lang ?? "en";
      let localVAD = 0.5;

      try {
        const stored = localStorage.getItem("eyeva-settings");
        if (stored) {
          const parsed = JSON.parse(stored);
          if (parsed.voice) localVoice = parsed.voice;
          if (parsed.speed) localSpeed = parsed.speed;
          if (parsed.ocrLang) localLang = parsed.ocrLang;
          if (parsed.vadThreshold !== undefined) localVAD = parsed.vadThreshold;
        }
      } catch {}

      // Send initial config
      const initialConfig = {
        type: "config",
        voice: localVoice,
        speed: localSpeed,
        lang: localLang,
        mode: optionsRef.current.mode ?? "voice",
        vad_threshold: localVAD
      };
      console.log("useVoice: Sending initial config via WS:", initialConfig);
      ws.send(JSON.stringify(initialConfig));
    };

    ws.onmessage = async (event) => {
      // Binary = WAV audio from TTS
      if (event.data instanceof Blob) {
        console.log("useVoice: WS received audio blob, size =", event.data.size);
        const arrayBuffer = await event.data.arrayBuffer();
        audioPlayerRef.current?.addChunk(arrayBuffer);
        return;
      }

      console.log("useVoice: WS received message text:", event.data);
      const msg = parseWSMessage(event.data as string);
      if (!msg) {
        console.warn("useVoice: Received unparsable message:", event.data);
        return;
      }

      switch (msg.type) {
        case "status":
          console.log("useVoice: WS state transition received ->", msg.state);
          updateStatus((msg.state as VoiceStatus) ?? "idle");
          break;

        case "transcript":
          if (msg.data) {
            console.log("useVoice: WS transcript received ->", msg.data);
            const entry: TranscriptEntry = {
              role: "user",
              text: msg.data as string,
              timestamp: Date.now(),
            };
            setTranscript((prev) => [...prev, entry]);
            optionsRef.current.onTranscript?.(msg.data as string);
          }
          break;

        case "response":
          if (msg.data) {
            console.log("useVoice: WS response received ->", msg.data);
            const entry: TranscriptEntry = {
              role: "ai",
              text: msg.data as string,
              timestamp: Date.now(),
            };
            setTranscript((prev) => [...prev, entry]);
            setLatestResponse(msg.data as string);
            optionsRef.current.onResponse?.(msg.data as string);
          }
          break;

        case "error":
          console.error("useVoice: WS error message received ->", msg.message);
          updateStatus("error");
          optionsRef.current.onError?.((msg.message as string) ?? "Unknown error");
          setTimeout(() => {
            console.log("useVoice: Resetting state to idle after WS error");
            updateStatus("idle");
          }, 3000);
          break;
      }
    };

    ws.onerror = (err) => {
      console.error(`useVoice: WebSocket connection error for URL: ${ws.url}. readyState: ${ws.readyState}`, err);
      setIsConnected(false);
      updateStatus("error");
      optionsRef.current.onError?.("Connection error. Reconnecting…");
    };

    ws.onclose = (event) => {
      console.log(`useVoice: WebSocket closed. code=${event.code}, reason=${event.reason || "none"}`);
      setIsConnected(false);
      updateStatus("idle");
      
      if (shouldReconnectRef.current) {
        if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = setTimeout(() => {
          console.log("useVoice: Reconnect timer fired");
          connect();
        }, 2000);
      }
    };
  }, [updateStatus]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        console.log("useVoice: Cleaning up WebSocket on unmount");
        // Prevent reconnect loop on clean close
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      // Clean up microphone stream and processor on unmount
      console.log("useVoice: Cleaning up microphone capture on unmount");
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect();
        audioProcessorRef.current.onaudioprocess = null;
        audioProcessorRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((t) => t.stop());
        audioStreamRef.current = null;
      }
    };
  }, [connect]);

  // Dynamically sync config changes with the active WebSocket session without reconnecting
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const updatedConfig = {
        type: "config",
        voice: options.voice ?? "af_heart",
        speed: options.speed ?? 1.0,
        lang: options.lang ?? "en",
        mode: options.mode ?? "voice"
      };
      console.log("useVoice: Config options changed, sending config via WS:", updatedConfig);
      wsRef.current.send(JSON.stringify(updatedConfig));
    }
  }, [options.voice, options.speed, options.lang, options.mode]);


  // Stop playback if status changes to listening (meaning user started talking/interrupted)
  useEffect(() => {
    if (status === "listening") {
      console.log("useVoice: Status transitioned to listening - stopping playback for interruption");
      audioPlayerRef.current?.stop();
    }
  }, [status]);




  // ── Stop Recording ───────────────────────────────────────────────────────────
  const stopListening = useCallback(() => {
    console.log("useVoice: stopListening requested");
    isRecordingRequestedRef.current = false;
    
    if (audioProcessorRef.current) {
      console.log("useVoice: Disconnecting audio processor node");
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current.onaudioprocess = null;
      audioProcessorRef.current = null;
    }
    if (audioContextRef.current) {
      console.log("useVoice: Closing AudioContext");
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (audioStreamRef.current) {
      console.log("useVoice: Stopping microphone stream tracks");
      audioStreamRef.current.getTracks().forEach((t) => t.stop());
      audioStreamRef.current = null;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log("useVoice: Sending transcribe command via WS (manual stop)");
      wsRef.current.send(JSON.stringify({ type: "command", action: "transcribe" }));
    } else {
      console.warn("useVoice: WS not open during stopListening manual trigger");
    }
  }, []);

  // ── Start Recording ──────────────────────────────────────────────────────────
  const startListening = useCallback(async () => {
    console.log("useVoice: startListening requested. Current status =", statusRef.current);
    // Interruption logic: Stop any currently playing audio when starting a new stream
    audioPlayerRef.current?.stop();

    if (audioProcessorRef.current || isRecordingRequestedRef.current) {
      console.warn("useVoice: Already recording or recording request already in progress");
      return;
    }

    isRecordingRequestedRef.current = true;
    try {
      console.log("useVoice: Querying audio input device permissions...");
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("navigator.mediaDevices or getUserMedia is not supported in this browser (likely due to insecure HTTP origin).");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;
      console.log("useVoice: Microphone stream obtained successfully");
      
      // If the request was cancelled while waiting for userMedia, abort
      if (!isRecordingRequestedRef.current) {
        console.warn("useVoice: startListening was cancelled during getUserMedia await, cleaning up tracks");
        stream.getTracks().forEach((t) => t.stop());
        audioStreamRef.current = null;
        return;
      }

      console.log("useVoice: Initializing Web Audio context with sampleRate = 16000Hz");
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      // Create a script processor with chunk size 512 samples
      const processor = ctx.createScriptProcessor(512, 1, 1);
      audioProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0); // Float32Array of raw PCM
        
        // Only stream raw PCM buffer if the assistant is not currently thinking (processing) or talking (speaking)
        // This prevents echo/loopback from the speaker from triggering the VAD/interruption loop.
        if (statusRef.current !== "processing" && statusRef.current !== "speaking") {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(inputData.buffer);
          }
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);

      console.log("useVoice: Web Audio raw PCM streaming started successfully");
      isRecordingRequestedRef.current = false;
      updateStatus("listening");
    } catch (err) {
      console.error("useVoice: startListening failed:", err);
      isRecordingRequestedRef.current = false;
      updateStatus("error");
      
      const errMsg = err instanceof Error ? err.message : String(err);
      optionsRef.current.onError?.(
        errMsg.includes("Permission denied") || errMsg.includes("getUserMedia is not supported")
          ? `Microphone error: ${errMsg}`
          : "Microphone access denied. Please enable microphone permissions."
      );
      
      setTimeout(() => {
        console.log("useVoice: Resetting state to idle after microphone error");
        updateStatus("idle");
      }, 3000);
    }
  }, [updateStatus]);

  // ── Send Camera Frame ────────────────────────────────────────────────────────
  const sendFrame = useCallback((frameB64: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "frame", data: frameB64 }));
    }
  }, []);

  // ── Update Config ─────────────────────────────────────────────────────────────
  const updateConfig = useCallback(
    (cfg: { voice?: string; speed?: number; lang?: string; mode?: string; vad_threshold?: number }) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "config", ...cfg }));
      }
    },
    []
  );

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    setLatestResponse("");
  }, []);

  // Start continuous streaming automatically on connection
  useEffect(() => {
    if (isConnected) {
      console.log("useVoice: WebSocket connected. Automatically starting continuous mic streaming.");
      startListening();
    }
  }, [isConnected, startListening]);

  return {
    status,
    transcript,
    latestResponse,
    isConnected,
    startListening,
    stopListening,
    sendFrame,
    updateConfig,
    clearTranscript,
  };
}
