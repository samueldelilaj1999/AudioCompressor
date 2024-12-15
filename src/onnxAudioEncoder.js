import * as ort from "onnxruntime-web";
import axios from "axios";

// Function to load the ONNX model and process the audio
export const encryptAudioWithOnnx = async (modelData, fileName) => {
  try {
    // Fetch the input tensor data
    const inputTensor = modelData;
    const modelPath = "/encodec_24Khz.onnx"; // Path to your model in the public folder

    const session = await ort.InferenceSession.create(modelPath);
    const output = await session.run({
      input_values: inputTensor,
    });

    const audioCodes = output.audio_codes;

    // Serialize tensor data to array for API compatibility
    const serializedAudioCodes = Array.from(audioCodes.data).map((item) =>
      item.toString()
    );

    // Send data to API
    await axios.post("http://localhost:8000/process-audio/", {
      audio_codes: serializedAudioCodes,
      dims: audioCodes.dims,
      file_name: fileName,
    });
  } catch (error) {
    throw new Error(
      "Error during ONNX inference or API call: " + error.message
    );
  }
};
