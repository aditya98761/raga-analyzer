/**
 * audio-worklet.js
 * AudioWorklet processor: captures mic PCM and posts Float32 chunks.
 *
 * Registered in app.js via:
 *   audioContext.audioWorklet.addModule('/static/audio-worklet.js')
 *
 * Sends chunks to main thread via this.port.postMessage({ pcm: Float32Array })
 * Main thread then sends them as binary over the WebSocket.
 */

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    // Accumulate samples until we have enough for a chunk
    this._chunkSize  = (options.processorOptions || {}).chunkSize || 2048;
    this._buf        = new Float32Array(this._chunkSize);
    this._bufPtr     = 0;

    this.port.onmessage = (e) => {
      if (e.data === 'stop') this._running = false;
    };
    this._running = true;
  }

  process(inputs) {
    if (!this._running) return false;

    const input   = inputs[0];
    const channel = input && input[0];
    if (!channel) return true;

    // Accumulate mono samples
    let srcIdx = 0;
    while (srcIdx < channel.length) {
      const toCopy = Math.min(channel.length - srcIdx, this._chunkSize - this._bufPtr);
      this._buf.set(channel.subarray(srcIdx, srcIdx + toCopy), this._bufPtr);
      this._bufPtr += toCopy;
      srcIdx       += toCopy;

      if (this._bufPtr >= this._chunkSize) {
        // Clone and send
        const chunk = new Float32Array(this._buf);
        this.port.postMessage({ pcm: chunk }, [chunk.buffer]);
        this._bufPtr = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);
