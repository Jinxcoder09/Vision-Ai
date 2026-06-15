/**
 * Eyeva AI — REST API Client
 * Typed wrappers around all backend REST endpoints.
 */

const isClient = typeof window !== "undefined";

const API_BASE = isClient
  ? `${window.location.protocol}//${window.location.host}`
  : (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000");

const WS_BASE = isClient
  ? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`
  : (process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000");

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  services: Record<string, boolean>;
}

export interface VisionResponse {
  description: string;
  cached: boolean;
  latency_ms: number;
}

export interface OCRResponse {
  text: string;
  word_count: number;
  lang: string;
  latency_ms: number;
}

export interface TranscribeResponse {
  transcript: string;
  language: string;
  confidence: number | null;
  latency_ms: number;
}

// ── Core Fetch Helper ────────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail ?? `API error: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── API Functions ─────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

export async function analyzeFrame(
  image_b64: string,
  prompt = "Describe the scene concisely for a visually impaired person."
): Promise<VisionResponse> {
  return apiFetch<VisionResponse>("/api/vision/analyze", {
    method: "POST",
    body: JSON.stringify({ image_b64, prompt }),
  });
}

export async function askQuestion(
  image_b64: string,
  question: string
): Promise<VisionResponse> {
  return apiFetch<VisionResponse>("/api/vision/question", {
    method: "POST",
    body: JSON.stringify({ image_b64, prompt: question }),
  });
}

export async function readText(
  image_b64: string,
  lang: "en" | "hi" = "en"
): Promise<OCRResponse> {
  return apiFetch<OCRResponse>("/api/ocr/read", {
    method: "POST",
    body: JSON.stringify({ image_b64, lang }),
  });
}

export async function transcribeAudio(
  audioBlob: Blob
): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  const res = await fetch(`${API_BASE}/api/speech/transcribe`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Transcription failed" }));
    throw new Error(err.detail);
  }
  return res.json();
}

export async function synthesizeSpeech(
  text: string,
  voice = "af_heart",
  speed = 1.0
): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}/api/speech/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed }),
  });
  if (!res.ok) {
    throw new Error("Speech synthesis failed");
  }
  return res.arrayBuffer();
}

// ── WebSocket Builders ────────────────────────────────────────────────────────

export function createSpeechWebSocket(): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/speech`);
}

export function createVisionWebSocket(): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/vision`);
}
