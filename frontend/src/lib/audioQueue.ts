/**
 * Eyeva AI – Frontend Audio Queue Player
 * Queues up and plays incoming WAV audio chunks sequentially to prevent overlapping.
 * Supports instant interruption / stopping of playback.
 */

export class AudioQueuePlayer {
  private queue: ArrayBuffer[] = [];
  private isPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;
  private audioCtx: AudioContext | null = null;

  /**
   * Adds an audio chunk (WAV) to the queue and starts playing if not already playing.
   */
  async addChunk(chunk: ArrayBuffer) {
    this.queue.push(chunk);
    if (!this.isPlaying) {
      await this.playNext();
    }
  }

  /**
   * Play the next item in the queue recursively.
   */
  private async playNext() {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const chunk = this.queue.shift()!;
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      }
      if (this.audioCtx.state === "suspended") {
        await this.audioCtx.resume();
      }

      // decodeAudioData consumes the ArrayBuffer, so we decode and play it
      const buffer = await this.audioCtx.decodeAudioData(chunk);
      const source = this.audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioCtx.destination);
      this.currentSource = source;

      await new Promise<void>((resolve) => {
        source.onended = () => {
          if (this.currentSource === source) {
            this.currentSource = null;
          }
          resolve();
        };
        source.start(0);
      });
    } catch (e) {
      console.warn("AudioQueuePlayer: Error playing audio chunk:", e);
    }

    // Process the next chunk in the queue
    await this.playNext();
  }

  /**
   * Stops all active playback, clears the queue, and resets states.
   */
  stop() {
    this.queue = [];
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch {
        // Source might have already stopped or not started
      }
      this.currentSource = null;
    }
    this.isPlaying = false;
  }
}
