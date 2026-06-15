// Utility function for merging class names
export function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(" ");
}

// Format latency for display
export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

// Truncate long text
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + "…";
}

// Convert audio blob to base64
export async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] ?? result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export function captureFrame(
  video: HTMLVideoElement,
  quality = 0.7
): string | null {
  const videoWidth = video.videoWidth || 1280;
  const videoHeight = video.videoHeight || 720;
  const aspectRatio = videoWidth / videoHeight;

  // Scale to 720p width (1280px) while maintaining the original aspect ratio
  const targetWidth = 1280;
  const targetHeight = Math.round(1280 / aspectRatio);

  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  return dataUrl.split(",")[1] ?? null;
}

// Play WAV audio bytes in browser
export async function playWavBytes(wavBytes: ArrayBuffer): Promise<void> {
  const ctx = new AudioContext();
  const buffer = await ctx.decodeAudioData(wavBytes);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(0);
  return new Promise((resolve) => {
    source.onended = () => {
      ctx.close();
      resolve();
    };
  });
}

// Parse WebSocket JSON message safely
export function parseWSMessage(raw: string): Record<string, unknown> | null {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
