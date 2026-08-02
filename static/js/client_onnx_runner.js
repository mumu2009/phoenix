/**
 * Phoenix v7.0 browser-side JS runner stub for server-client model deployment.
 *
 * In server-client mode the vision/speech models are executed in the client
 * (browser, Electron, or Node.js/WASM runtime).  The client is responsible for:
 *   - preprocessing image/audio into the model's input tensors,
 *   - running the additive residual ONNX model (via onnxruntime-web or a WASM
 *     build),
 *   - sending the resulting concept vector to the Phoenix backend,
 *   - receiving a concept vector from the backend and decoding it back to an
 *     image or audio payload.
 *
 * This file is a minimal stub and documentation target.  Production
 * implementations should replace the placeholder functions with real
 * onnxruntime-web inference or a WebGPU/WebAssembly runner.
 */

const ClientOnnxRunner = {
  /**
   * Encode an image payload to a concept vector using a client-side ONNX model.
   *
   * @param {Uint8Array} imageBytes  raw image bytes (PNG/JPEG/BGR).
   * @param {number} width           image width in pixels.
   * @param {number} height          image height in pixels.
   * @param {string} mimeType        e.g. "image/png" or "application/x-bgr".
   * @param {string} modelUrl        URL of the vision_encoder ONNX.
   * @returns {Promise<Float32Array>} concept vector.
   */
  async encodeImage(imageBytes, width, height, mimeType, modelUrl) {
    // TODO: load `modelUrl` with onnxruntime-web, preprocess imageBytes into
    // NCHW [1, 3, 224, 224], run inference, and return the flat concept output.
    console.log("[ClientOnnxRunner.encodeImage] stub", { imageBytes, width, height, mimeType, modelUrl });
    throw new Error("Client-side image ONNX inference is not implemented in this stub");
  },

  /**
   * Decode a concept vector back to an image payload.
   *
   * @param {Float32Array} conceptVector  concept vector from the backend.
   * @param {string} mimeType             desired output MIME ("image/png" or "image/jpeg").
   * @param {string} modelUrl             URL of the vision_decoder ONNX.
   * @returns {Promise<Uint8Array>}       encoded image bytes.
   */
  async decodeImage(conceptVector, mimeType, modelUrl) {
    console.log("[ClientOnnxRunner.decodeImage] stub", { conceptVector, mimeType, modelUrl });
    throw new Error("Client-side image decode ONNX inference is not implemented in this stub");
  },

  /**
   * Encode an audio payload to a concept vector using a client-side ONNX model.
   *
   * @param {Uint8Array} audioBytes  raw audio bytes (PCM or WAV).
   * @param {number} sampleRate      sample rate in Hz.
   * @param {string} mimeType        e.g. "audio/pcm" or "audio/wav".
   * @param {string} modelUrl        URL of the speech_encoder ONNX.
   * @returns {Promise<Float32Array>} concept vector.
   */
  async encodeAudio(audioBytes, sampleRate, mimeType, modelUrl) {
    console.log("[ClientOnnxRunner.encodeAudio] stub", { audioBytes, sampleRate, mimeType, modelUrl });
    throw new Error("Client-side audio ONNX inference is not implemented in this stub");
  },

  /**
   * Decode a concept vector back to an audio payload.
   *
   * @param {Float32Array} conceptVector  concept vector from the backend.
   * @param {string} mimeType             desired output MIME ("audio/pcm" or "audio/wav").
   * @param {number} lengthHint           requested output length in samples.
   * @param {string} modelUrl             URL of the speech_decoder ONNX.
   * @returns {Promise<Uint8Array>}       encoded audio bytes.
   */
  async decodeAudio(conceptVector, mimeType, lengthHint, modelUrl) {
    console.log("[ClientOnnxRunner.decodeAudio] stub", { conceptVector, mimeType, lengthHint, modelUrl });
    throw new Error("Client-side audio decode ONNX inference is not implemented in this stub");
  },

  /**
   * Build a JSON concept payload for the Phoenix server-client REST endpoint.
   *
   * @param {Float32Array} conceptVector  client-computed concept vector.
   * @param {string} modality             "image" or "audio".
   * @returns {object}                    JSON body for /api/mixed_modal/encode.
   */
  buildConceptPayload(conceptVector, modality) {
    return {
      modality,
      conceptVector: Array.from(conceptVector),
      source: "client",
    };
  },
};

// Export for both CommonJS and browser globals.
if (typeof module !== "undefined" && module.exports) {
  module.exports = ClientOnnxRunner;
} else if (typeof window !== "undefined") {
  window.ClientOnnxRunner = ClientOnnxRunner;
}
