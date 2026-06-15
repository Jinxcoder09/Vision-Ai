"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { captureFrame } from "@/lib/utils";

interface UseCameraOptions {
  deviceId?: string;
  onError?: (err: string) => void;
}

interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isActive: boolean;
  error: string | null;
  devices: MediaDeviceInfo[];
  startCamera: (deviceId?: string) => Promise<void>;
  stopCamera: () => void;
  captureCurrentFrame: (quality?: number) => string | null;
  switchCamera: (deviceId: string) => Promise<void>;
}

export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  // Store options in a ref to avoid recreation of hooks
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Enumerate camera devices
  const loadDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === "videoinput"));
    } catch {
      // permissions not yet granted — will be populated after startCamera
    }
  }, []);

  useEffect(() => {
    loadDevices();
  }, [loadDevices]);

  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      setError(null);

      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      let activeDeviceId = deviceId;
      let targetWidth = 1280;
      let targetHeight = 720;
      let targetFps = 30;

      try {
        const stored = localStorage.getItem("eyeva-settings");
        if (stored) {
          const parsed = JSON.parse(stored);
          if (!activeDeviceId && parsed.cameraId) {
            activeDeviceId = parsed.cameraId;
          }
          if (parsed.resolution) {
            if (parsed.resolution === "1080p") {
              targetWidth = 1920;
              targetHeight = 1080;
            } else if (parsed.resolution === "480p") {
              targetWidth = 854;
              targetHeight = 480;
            }
          }
          if (parsed.fps) {
            targetFps = parsed.fps;
          }
        }
      } catch {}

      let stream: MediaStream;
      try {
        const videoConstraints: MediaTrackConstraints = activeDeviceId
          ? { deviceId: { exact: activeDeviceId } }
          : { facingMode: "environment" };

        videoConstraints.width = { ideal: targetWidth };
        videoConstraints.height = { ideal: targetHeight };
        videoConstraints.frameRate = { ideal: targetFps };

        const constraints: MediaStreamConstraints = {
          video: videoConstraints,
          audio: false,
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr) {
        console.warn("useCamera: Ideal constraints failed, attempting fallback constraints...", firstErr);
        const fallbackConstraints: MediaStreamConstraints = {
          video: activeDeviceId ? { deviceId: { exact: activeDeviceId } } : true,
          audio: false,
        };
        stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsActive(true);
      await loadDevices(); // refresh after permission granted
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Camera unavailable.";
      const userMsg =
        msg.includes("NotAllowedError") || msg.includes("Permission")
          ? "Camera permission denied. Please allow camera access."
          : msg.includes("NotFoundError") || msg.includes("DevicesNotFound")
          ? "No camera found. Please connect a camera."
          : "Camera connection failed. Please try again.";
      setError(userMsg);
      setIsActive(false);
      optionsRef.current.onError?.(userMsg);
    }
  }, [loadDevices]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
  }, []);

  const captureCurrentFrame = useCallback(
    (quality = 0.7): string | null => {
      if (!videoRef.current || !isActive) return null;
      return captureFrame(videoRef.current, quality);
    },
    [isActive]
  );

  const switchCamera = useCallback(
    async (deviceId: string) => {
      await startCamera(deviceId);
    },
    [startCamera]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    isActive,
    error,
    devices,
    startCamera,
    stopCamera,
    captureCurrentFrame,
    switchCamera,
  };
}
