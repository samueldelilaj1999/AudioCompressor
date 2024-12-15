import { FFmpeg } from "@ffmpeg/ffmpeg";
import * as onnx from "onnxruntime-web";

export const convertTo24kHz = async (file) => {
  if (!file) return null;

  const ffmpeg = new FFmpeg({ log: false });
  await ffmpeg.load();

  try {
    const fileArrayBuffer = await file.arrayBuffer();
    const uint8Array = new Uint8Array(fileArrayBuffer);

    await ffmpeg.writeFile(file.name, uint8Array);

    await ffmpeg.exec([
      "-i",
      file.name,
      "-ar",
      "24000",
      "-ac",
      "1",
      "-c:a",
      "pcm_f32le",
      "output.wav",
    ]);

    const data = await ffmpeg.readFile("output.wav");

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(data.buffer);

    const originalSampleRate = audioBuffer.sampleRate;
    const targetSampleRate = 24000;
    const offlineCtx = new OfflineAudioContext(
      1,
      (audioBuffer.length * targetSampleRate) / originalSampleRate,
      targetSampleRate
    );
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);
    const resampledBuffer = await offlineCtx.startRendering();

    const channelData = resampledBuffer.getChannelData(0);
    let max = 0;
    for (let i = 0; i < channelData.length; i++) {
      const absValue = Math.abs(channelData[i]);
      if (absValue > max) {
        max = absValue;
      }
    }
    const normalizedArray = channelData.map((sample) => sample / (max || 1));

    const tensorData = new onnx.Tensor("float32", normalizedArray, [
      1,
      1,
      normalizedArray.length,
    ]);

    return tensorData;
  } catch (error) {
    console.error("Conversion error:", error);
    throw error;
  }
};
