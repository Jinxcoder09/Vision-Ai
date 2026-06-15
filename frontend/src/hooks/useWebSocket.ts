"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { parseWSMessage } from "@/lib/utils";

type WSStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseWebSocketOptions {
  url: string;
  onMessage?: (msg: Record<string, unknown>) => void;
  onBinaryMessage?: (data: ArrayBuffer) => void;
  onStatusChange?: (status: WSStatus) => void;
  reconnectDelay?: number;
  autoConnect?: boolean;
}

interface UseWebSocketReturn {
  status: WSStatus;
  sendJSON: (payload: Record<string, unknown>) => void;
  sendBytes: (data: ArrayBuffer | Blob) => void;
  disconnect: () => void;
  connect: () => void;
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldReconnectRef = useRef(true);
  const [status, setStatus] = useState<WSStatus>("disconnected");

  // Store options in a ref to avoid stale closures
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const updateStatus = useCallback(
    (s: WSStatus) => {
      setStatus(s);
      optionsRef.current.onStatusChange?.(s);
    },
    []
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (!optionsRef.current.url) return;

    updateStatus("connecting");
    const ws = new WebSocket(optionsRef.current.url);
    wsRef.current = ws;

    ws.onopen = () => updateStatus("connected");

    ws.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        const buf = await event.data.arrayBuffer();
        optionsRef.current.onBinaryMessage?.(buf);
        return;
      }
      const msg = parseWSMessage(event.data as string);
      if (msg) optionsRef.current.onMessage?.(msg);
    };

    ws.onerror = () => updateStatus("error");

    ws.onclose = () => {
      updateStatus("disconnected");
      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = setTimeout(
          connect,
          optionsRef.current.reconnectDelay ?? 2000
        );
      }
    };
  }, [updateStatus]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    wsRef.current?.close();
  }, []);

  const sendJSON = useCallback((payload: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  const sendBytes = useCallback((data: ArrayBuffer | Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    if (options.autoConnect !== false) {
      shouldReconnectRef.current = true;
      connect();
    }
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Mount-only effect — runs once on mount, cleans up on unmount

  return { status, sendJSON, sendBytes, disconnect, connect };
}
